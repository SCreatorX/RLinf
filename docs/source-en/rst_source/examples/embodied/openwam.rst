OpenWAM Reinforcement Learning
==============================

This example plugs the OpenWAM video world model into RLinf's PPO loop: the HuggingFace rollout worker generates actions with OpenWAM's native joint inference and caches the denoising chain, the FSDP actor replays that chain exactly to compute differentiable log-probabilities and values, and the trained weights can be exported back to a native OpenWAM checkpoint directory for deployment and evaluation. See :doc:`sft_openwam` for supervised fine-tuning.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      OpenWAM (Wan2.2 dual-system; other architectures below)

   .. grid-item-card:: Method
      :text-align: center

      PPO (denoising-transition surrogate likelihood)

   .. grid-item-card:: Environment
      :text-align: center

      LIBERO spatial / object / goal / 10

   .. grid-item-card:: Hardware
      :text-align: center

      8 GPUs: 2 actor, 1 rollout, the rest for rendering

Design notes
------------

- **Execute a prefix of each chunk, not the whole chunk.** The model predicts ``num_frames - 1 = 32`` actions per generation, but OpenWAM's deploy executor executes only ``inference_horizon = 10`` before regenerating. Executing all 32 open-loop scores 0 on LIBERO-Spatial; executing 10 solves the task in about 80 steps. Set ``openwam.inference_horizon: 10`` and ``num_action_chunks: 10`` on both ``rollout.model`` and ``actor.model``; behaviour and actor log-probabilities cover the executed 10 frames only.
- **Match the action representation to the checkpoint.** ``new-openwam-libero-sft-10epoch-delta-aligned`` predicts EEF10 deltas in OSC command units and needs ``openwam_action_representation: native_delta_eef10``; the older ``openwam-libero-sft-30000`` predicts absolute end-effector targets and needs ``absolute_eef10``.
- **Differentiable replay.** ``predict_action_batch`` samples one denoising transition with Gaussian exploration and packs OpenWAM's non-flat native conditioning (Cosmos ``und_kv``, tri-system VLM features, ...) into fixed-shape tensors with ``rlinf.models.embodiment.openwam.replay.pack_native_inputs``; the actor's ``default_forward`` unpacks and replays it per sample, so on unchanged weights ``actor/ratio`` is exactly 1.0 and ``actor/approx_kl`` is 0.
- **Value head.** With ``actor.model.add_value_head: true`` RLinf attaches a small MLP over pooled statistics of the video latents, text context, proprioception and action. It is saved in FSDP checkpoints, exported as ``rlinf_value_head.pt``, and reloaded automatically when a run resumes from an export.

Supported architectures
-----------------------

The PPO bridge dispatches on the checkpoint's architecture, so any OpenWAM architecture whose ``forward(noisy_actions, action_timestep, proprio=..., **pipeline_inputs)`` returns an action flow prediction can be trained. ``toolkits/openwam/check_ppo_replay.py`` runs the model-level check (rollout, trajectory transport, actor rescoring, backward) for one checkpoint and writes the metrics as JSON.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Architecture / backbone
     - Status
   * - dual_system ``joint_self_attn`` (Wan2.2 TI2V 5B)
     - validated end to end; used by all LIBERO PPO runs
   * - dual_system ``joint_cross_attn``, ``idm``
     - validated at model level
   * - single_system ``vanilla``, ``moe``
     - validated at model level
   * - tri_system ``joint_self_attn`` (Qwen3-VL)
     - validated at model level; the VLM must stay frozen, its hidden states are computed once per observation and replayed
   * - Wan2.1 VACE 1.3B, Wan2.1 I2V 14B
     - validated at model level; a single 14B replay sample peaks at about 117 GB, use FSDP across actor ranks
   * - Cosmos3 Edge
     - validated at model level; ``unify_action`` scatters the 20-D physical state into an 80-D unified vector
   * - Cosmos-Predict2.5
     - untested; needs OpenWAM's ``install_cosmos_predict25.sh`` extras

Installation
------------

.. include:: _setup_common.rst

.. code-block:: bash

   bash requirements/install.sh embodied --model openwam --env libero
   source .venv/bin/activate

LIBERO's EGL rendering needs GLVND's ``libEGL.so.1`` (``apt-get install libegl1`` on Debian/Ubuntu) and ``MUJOCO_GL=egl``, ``PYOPENGL_PLATFORM=egl``.

Running
-------

Config files
~~~~~~~~~~~~

- ``examples/embodiment/config/libero_spatial_ppo_openwam.yaml``: the full recipe (64 environments, 8 rollout epochs).
- ``examples/embodiment/config/libero_spatial_ppo_openwam_long.yaml``: sustained validation (8 environments, 320-step episodes, 4 denoising steps, 20 iterations).
- ``examples/embodiment/config/libero_spatial_ppo_openwam_smoke.yaml`` and ``tests/e2e_tests/embodied/libero_spatial_ppo_openwam.yaml``: smoke recipes.
- ``examples/embodiment/config/robotwin_click_bell_ppo_openwam.yaml``: PPO on RoboTwin click_bell (aloha-agilex, 20-D absolute EEF, whole 32-step chunks, 4 environments and 52 samples per iteration); the env adapter is described in :doc:`../../evaluations/guides/robotwin`. Not yet exercised in a live RoboTwin environment.

Key snippet:

.. code-block:: yaml

   env:
     train:
       openwam_action_representation: native_delta_eef10
       total_num_envs: 4
       render_gpu_ids: [4, 5, 6, 7]   # one render GPU per LIBERO subprocess, never shared with CUDA work
       task_id_filter: [2, 3, 4]      # optional: train on a subset of tasks

   rollout:
     model:
       model_type: openwam
       model_path: /path/to/new-openwam-libero-sft-10epoch-delta-aligned
       num_frames: 33
       denoise_steps: 4
       num_action_chunks: 10
       openwam:
         inference_horizon: 10

   actor:
     model:
       model_path: ${rollout.model.model_path}
       add_value_head: true
       openwam:
         rl_enabled: true
         noise_std: 0.05
         inference_horizon: 10
     micro_batch_size: 1
     global_batch_size: 128

Batch-size constraints
~~~~~~~~~~~~~~~~~~~~~~

Each iteration collects ``total_num_envs * rollout_epoch * (max_steps_per_rollout_epoch / num_action_chunks)`` action-chunk samples. That number must be divisible by ``actor.global_batch_size``, which in turn must be divisible by ``actor.micro_batch_size * actor_world_size``; RLinf checks this before loading any model. A ``global_batch_size`` smaller than the rollout means several optimizer steps per iteration, and later micro-batches are scored against an already-updated policy, so KL rises; prefer one or two updates per iteration until a learning signal is established.

GPU placement
~~~~~~~~~~~~~

Several LIBERO subprocesses rendering every step on the same EGL device abort natively in ``read_pixels`` (the parent sees ``EOFError``, ``exitcode=-6``). ``env.<split>.render_gpu_ids`` pins each subprocess renderer to a GPU (``MUJOCO_EGL_DEVICE_ID``, round-robin), and those GPUs must not run CUDA work. On an 8-GPU node with the actor on 0-1, the rollout on 3 and the env worker on 2, that leaves four render GPUs, i.e. four to six environments.

Launch
~~~~~~

.. code-block:: bash

   export PYTHONPATH=$PWD EMBODIED_PATH=$PWD/examples/embodiment
   export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   python examples/embodiment/train_embodied_agent.py \
       --config-name libero_spatial_ppo_openwam_long \
       env.train.total_num_envs=4 +env.train.render_gpu_ids=[4,5,6,7] \
       actor.global_batch_size=128 runner.max_epochs=30 runner.max_steps=30 runner.save_interval=10

Metrics to watch
~~~~~~~~~~~~~~~~

- ``rollout/success_once``, ``rollout/return``: per-iteration success and return, the only direct signal of policy improvement.
- ``actor/ratio``, ``actor/approx_kl``, ``actor/clip_fraction``: ratio 1.0 and KL 0 on unchanged weights; the tabulated ``actor/ratio`` is an unmasked mean and tracks ``loss_mask_fraction``.
- ``critic/explained_variance``: the current value head sees only eight pooled scalars, so explained variance stays near 0 and advantages degrade to normalized returns. This is a known limitation, not a training fault.
- ``rollout/rewards``, ``rollout/returns_max``: actor-side masked means, computed per actor rank and averaged.

Export and evaluation
---------------------

The FSDP actor stores the whole ``OpenWAMPolicy`` state dict at ``<log_path>/<experiment>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt``. Rebuild a native OpenWAM checkpoint directory from it:

.. code-block:: bash

   python toolkits/openwam/export_ppo_checkpoint.py \
       --rlinf-checkpoint <log_path>/<experiment>/checkpoints/global_step_20 \
       --source-checkpoint /path/to/new-openwam-libero-sft-10epoch-delta-aligned \
       --output /path/to/openwam-libero-ppo-step20 --link-assets --verify cuda

The exporter strips the ``architecture.`` prefix, drops ``vlm_backbone.*`` (OpenWAM stores the VLM as a directory), checks that the key set equals the source checkpoint, writes ``checkpoint_step_N.safetensors`` and saves the value head as ``rlinf_value_head.pt``; ``--verify`` reloads the result through ``openwam.deploy.load_from_checkpoint_dir``. The export directory can be passed straight to an evaluation recipe as ``rollout.model.model_path``:

.. code-block:: bash

   python evaluations/eval_embodied_agent.py --config-name libero_spatial_openwam_eval \
       rollout.model.model_path=/path/to/openwam-libero-ppo-step20 \
       env.eval.total_num_envs=5 env.eval.render_gpu_ids=[2,3,4,5,6] env.eval.rollout_epoch=10

``evaluations/libero/`` ships ``libero_{spatial,object,goal,10}_openwam_eval.yaml``; every suite keeps the 600-step cap of OpenWAM's native LIBERO client. Each episode is logged as ``[libero eval] task_id=.., trial_id=.., success=..``, so success can be split per task.

Current results
---------------

LIBERO-Spatial success on the same checkpoint lineage and the same trials (10 tasks x 5 trials, 5 environments, 4 denoising steps, horizon 10):

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Checkpoint
     - Tasks 2/3/4 (trained on)
     - Other 7 tasks
     - Total
   * - SFT baseline
     - 4/15
     - 32/35
     - 36/50
   * - PPO step 10 (tasks 2/3/4, 4 envs, 30 iterations)
     - 5/15
     - 33/35
     - 38/50
   * - PPO step 20
     - 5/15
     - 31/35
     - 36/50
   * - PPO step 30
     - 2/15
     - 33/35
     - 35/50

At five trials per task PPO shows no measurable gain and no regression on the untrained tasks. Eight trajectories per iteration and an uninformative value head are the known sample-efficiency bottlenecks.

Known limitations
-----------------

- The video stream is deterministic conditioning in this PPO port; the action likelihood is a denoising-transition surrogate, not the exact marginal after integrating over video trajectories.
- Only the Wan2.2 dual-system checkpoint has run full PPO on LIBERO; the other architectures are validated at model level only and Cosmos-Predict2.5 is untested.
- Full PPO has only run on LIBERO. The RoboTwin evaluation adapter is wired (``evaluations/robotwin/robotwin_{click_bell,place_empty_cup}_openwam_eval.yaml``, see :doc:`../../evaluations/guides/robotwin`) but has not been exercised in a live RoboTwin environment yet.
- Multiple LIBERO renderers on one GPU crash, which bounds the number of environments per node.
