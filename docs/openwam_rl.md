# OpenWAM PPO integration

`libero_spatial_ppo_openwam.yaml` runs an OpenWAM checkpoint through RLinf's HuggingFace rollout and FSDP actor. Rollout mode uses the native joint `architecture.forward` and records a short flow-matching action chain. One transition is sampled with Gaussian exploration; the actor replays that transition to compute differentiable log-probabilities and a value from pooled video/context/proprio features.

The video stream is deterministic conditioning in this first PPO port. This makes the action likelihood an explicit denoising-transition surrogate, rather than an exact marginal likelihood of the final action after integrating out the video trajectory.

## Checkpoint and execution settings that matter

Two settings decide whether an OpenWAM checkpoint does anything useful in LIBERO; both were found by running the same checkpoint through OpenWAM's own LIBERO client and through RLinf on identical observations:

- **Execute a prefix of each chunk.** The model predicts `num_frames - 1 = 32` actions per generation, but OpenWAM's deploy executor executes only `inference_horizon = 10` of them before regenerating. Executing all 32 open-loop scores 0/2 on LIBERO-Spatial (episodes time out), while executing 10 succeeds in about 80 steps, in a plain LIBERO env and through RLinf's `LiberoEnv` alike. Set `openwam.inference_horizon: 10` on both `rollout.model` and `actor.model`, and `num_action_chunks: 10`. In PPO the executed prefix is the RL action: the behaviour and actor log-probabilities cover those 10 frames only, while the replay inputs keep the full sampled chain.
- **Use a checkpoint whose action representation matches the bridge.** `new-openwam-libero-sft-10epoch-delta-aligned` predicts native EEF10 deltas in OSC command units (its `normalization_stats.npy` carries an `eef` block for actions and an `eef_state` block for proprio) and reaches 96 to 100 percent on the natively evaluated LIBERO tasks; use it with `openwam_action_representation: native_delta_eef10`. The older `openwam-libero-sft-30000` predicts absolute EEF targets (a single `eef` block with metre-scale positions) and needs `absolute_eef10`, which converts the target into a per-step OSC command against the current pose; it has no native LIBERO evaluation showing it works.

`toolkits/openwam/check_ppo_replay.py` covers the model side of the bridge; the two facts above were established with one-off scripts that fed a LIBERO observation through both the native client (`benchmarks/libero/openwam2libero_interface.py`) and `OpenWAMPolicy`: composite images differ only by resampling filter, proprio and converted actions agree to float precision.

Set `actor.model.model_path` and `rollout.model.model_path` to the same deploy checkpoint. The actor needs `actor.model.add_value_head: true`; the value head is initialized by RLinf and included in FSDP checkpoints. Keep `algorithm.recompute_logprobs: false`: rollout stores the sampled transition log-probability and the actor-side `default_forward` recomputes it from the cached native tensors.

## Supported architectures and backbones

The PPO bridge dispatches on the checkpoint's architecture, so every OpenWAM architecture whose `forward(noisy_actions, action_timestep, proprio=..., **pipeline_inputs)` returns an action flow prediction can be trained. Model-level replay checks (rollout, RLinf-style trajectory transport, actor rescoring with `ratio == 1.0` on unchanged weights, finite gradients through the action path) have been run on the `OpenWAM_Study_Checkpoints`:

| Architecture / backbone | Status |
| --- | --- |
| dual_system `joint_self_attn` (Wan2.2 TI2V 5B) | validated; used by all LIBERO PPO runs |
| dual_system `joint_cross_attn`, `idm` | validated at model level |
| single_system `vanilla`, `moe` | validated at model level |
| tri_system `joint_self_attn` (Qwen3-VL) | validated at model level; the frozen VLM is run once per observation and its hidden states are replayed |
| Wan2.1 VACE 1.3B, Wan2.1 I2V 14B | validated at model level (a single 14B replay sample peaks at about 117 GB on one GPU; use FSDP across actor ranks) |
| Cosmos3 Edge | validated at model level (trained with `unify_action`: the 20-D physical state is scattered into an 80-D unified vector by the checkpoint's normalizer) |
| Cosmos-Predict2.5 | untested: requires OpenWAM's `install_cosmos_predict25.sh` extras |

`toolkits/openwam/check_ppo_replay.py` runs this check for one checkpoint and writes the metrics as JSON; use it before enabling PPO on a checkpoint family that is not listed above.

A tri-system checkpoint must keep its VLM frozen (`requires_grad=False`, the OpenWAM default); PPO raises otherwise because the cached VLM features would go stale.

### Native conditioning replay

RLinf carries rollout `forward_inputs` as flat tensors: they are split per environment on the batch axis, stacked over time, flattened, and shuffled before the actor sees them. OpenWAM's native conditioning is not flat (Cosmos `und_kv` is a list of key/value tuples, tri-system carries VLM features, several entries are scalars or `None`), so the policy packs each observation with `rlinf.models.embodiment.openwam.replay.pack_native_inputs` into fixed-shape tensors plus a schema tensor, and unpacks them per sample at replay time. Text-length axes (`context`, `context_mask`, `und_kv`, `vlm_hidden`, ...) are padded for transport to `actor.model.openwam.replay_text_capacity` (default 512, matching the Qwen3-VL and Cosmos3 tokenizer limits) and trimmed back before the architecture sees them.

### Proprioception

LIBERO observations expose `states`; the bridge converts them to OpenWAM's 10-D absolute EEF representation. An environment can instead provide `native_proprio` (a finite vector in the checkpoint's physical state units, e.g. the 20-D Robotwin state); it is normalized with the checkpoint's `normalization_stats.npy` and bypasses the LIBERO conversion. `RoboTwinEnv` does this when `env.<split>.openwam_action_representation: absolute_eef20` is set: it reads the two end-effector poses and gripper openings from each RoboTwin task (`[l_xyz, l_rot6d, l_grip, r_xyz, r_rot6d, r_grip]`, the dataset's `endpose` layout), binds `action_type="ee"` on the tasks' `gen_sparse_reward_data`, and `prepare_actions` converts the model's 20-D EEF chunk into RoboTwin's 16-D `xyz+quat_xyzw+gripper` command. Checkpoints trained by a RoboTwin-style reader also get its instruction prefix (`format_prompt_for_inference`) prepended to the task description, and the head plus both wrist cameras fill the three `camera_layout` slots. Checkpoints trained with `dataloader.unify_action` report `proprio_dim`/`action_dim` of the unified width (80); the physical width is the normalizer's, and `active_action_indices` selects the physical action dims for the PPO log-probability.

## Exporting a PPO checkpoint for OpenWAM eval

The FSDP actor saves `<log_path>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt` (the whole `OpenWAMPolicy` state dict) next to the sharded DCP checkpoint. OpenWAM's deploy/eval tooling reads self-contained checkpoint directories, so rebuild one with:

```bash
python toolkits/openwam/export_ppo_checkpoint.py \
    --rlinf-checkpoint ../results/libero_spatial_ppo_openwam/checkpoints/global_step_20 \
    --source-checkpoint /path/to/openwam-libero-sft-30000 \
    --output /path/to/openwam-libero-ppo-step20 --verify cuda
```

The exporter strips the `architecture.` prefix, drops `vlm_backbone.*` (OpenWAM stores the VLM as a directory), checks that the key set equals the source `checkpoint_step_*.safetensors`, writes `checkpoint_step_N.safetensors`, copies `config.yaml`, `normalization_stats.npy`, tokenizer and VLM directories from the source checkpoint (`--link-assets` symlinks them instead), and keeps the PPO value head in `rlinf_value_head.pt`. When `actor.model.model_path` (or `rollout.model.model_path`) points at such an export, `OpenWAMPolicy.from_checkpoint` reloads that file into the value head, so a PPO run resumed from an export keeps its critic; pass `load_value_head=False` to start from a fresh head. `--verify` reloads the result through `openwam.deploy.load_from_checkpoint_dir`. Sharded-only checkpoints (`save_full_model_weights: false`) are consolidated through `torch.distributed.checkpoint` first.

## Batch sizing and validation runs

The number of action-chunk samples per iteration is `total_num_envs * rollout_epoch * (max_steps_per_rollout_epoch / num_action_chunks)`. It must be divisible by `actor.global_batch_size`, which must in turn be divisible by `actor.micro_batch_size * actor_world_size`. OpenWAM PPO validates these constraints before loading models or collecting trajectories. With the 10-step horizon the larger recipe collects 16384 samples and uses global batch 512; the sustained-validation recipe collects 256 samples and updates once per iteration (global batch 256). Both use micro batch 1 with gradient accumulation to limit video-model activation memory.

Rollout scores the behavior log-probability one observation at a time and replays the exact sampled timesteps, so on unchanged weights `actor/ratio` is 1.0 and `actor/approx_kl` is 0. With several optimizer steps per rollout (`global_batch_size` smaller than the rollout), later micro-batches are scored against an already-updated policy: in the 8-environment sustained run, global batch 16 produced a first-iteration KL near 9 and clip fraction near 0.4, while one update per rollout (global batch 80) kept KL at 0. Treat large KL under multi-update iterations as a batch-size choice, and prefer fewer, larger updates or a KL-based early stop until a learning signal has been established.

For sustained validation, use `--config-name libero_spatial_ppo_openwam_long`. It retains 320-step LIBERO episodes and four denoising steps, while collecting from eight environments for one rollout epoch before each actor phase. It runs 20 iterations and saves at iterations 10 and 20. This configuration is intended to test repeated updates and reward trends; successful completion alone does not establish policy improvement. The runs before the horizon fix executed full 32-step chunks from the absolute-EEF checkpoint and scored 0 reward throughout, so they verified the training loop and nothing about policy improvement.

The current OpenWAM rollout processes environments sequentially within `predict_action_batch`. In the 64-environment test, eight rollout epochs completed in 69 minutes before actor training began. Two, four and eight environments have been validated end to end; a ten-environment run failed in the LIBERO subprocesses (`exitcode=-6`), and the default 64-environment recipe has not been validated through an actor update. With the 10-step horizon a single-environment eval runs to completion through the full RLinf pipeline (`eval/success_once = 1.0`, task solved in about 80 steps). Runs with two or more environments lose LIBERO subprocesses to a native abort in `robosuite ... read_pixels` (`mjr_readPixels`; the parent sees `EOFError`, `exitcode=-6`, and `PYTHONFAULTHANDLER=1` shows the stack). A model-free reproducer isolates it: two `LiberoEnv` subprocesses rendering every step on the same EGL device abort at the first step, every time; the same two subprocesses pinned to different GPUs, or a single subprocess, or two subprocesses rendering only at chunk boundaries (`skip_intermediate_renders: true`), all run. Policy-driven runs survive longer (12 to 255 generations) because the model call spaces the renders out, but they end the same way.

`env.<split>.render_gpu_ids` pins each subprocess renderer to a GPU (`MUJOCO_EGL_DEVICE_ID`, round-robin). Give every renderer its own device and keep those devices free of CUDA work: a pinned run whose renderer shared a GPU with the rollout model still aborted after 28 generations, which matches the warning in OpenWAM's LIBERO README. On an 8-GPU node with the actor and rollout on two GPUs this caps a LIBERO run at about six environments until the same-device abort is fixed in the renderer stack; the eval recipe therefore runs three environments on GPUs 0, 2 and 3 with the model on GPU 1. Reward and success rate must be checked alongside losses: PPO loss is not expected to decrease monotonically across fresh on-policy batches.

`tests/e2e_tests/embodied/libero_spatial_ppo_openwam.yaml` is the CI-sized recipe (two environments, 60-step episodes, two iterations, two denoising steps); run it with `bash tests/e2e_tests/embodied/run.sh libero_spatial_ppo_openwam`.
