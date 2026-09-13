# OpenWAM PPO integration

`libero_spatial_ppo_openwam.yaml` runs the OpenWAM Wan22 dual-system checkpoint through RLinf's HuggingFace rollout and FSDP actor. Rollout mode uses the native joint `architecture.forward` and records a short flow-matching action chain. One transition is sampled with Gaussian exploration; the actor replays that transition to compute differentiable log-probabilities and a value from pooled video/context/proprio features.

The video stream is deterministic conditioning in this first PPO port. This makes the action likelihood an explicit denoising-transition surrogate, rather than an exact marginal likelihood of the final action after integrating out the video trajectory. The implementation therefore currently targets the Wan22 dual-system architecture already validated for OpenWAM SFT/eval. Other OpenWAM backbones remain eval/SFT-compatible and should not be enabled for PPO until their `forward` signatures and action scheduler are validated.

Set `actor.model.model_path` and `rollout.model.model_path` to the same deploy checkpoint. The actor needs `actor.model.add_value_head: true`; the value head is initialized by RLinf and included in FSDP checkpoints. Keep `algorithm.recompute_logprobs: false`: rollout stores the sampled transition log-probability and the actor-side `default_forward` recomputes it from the cached native tensors.
