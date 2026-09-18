# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLinf policy wrapper around OpenWAM's native deployment engine."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.openwam.replay import (
    pack_native_inputs,
    unpack_native_inputs,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


EXPORTED_VALUE_HEAD_FILE = "rlinf_value_head.pt"


def load_exported_value_head(model_path: str, value_head: nn.Module) -> bool:
    """Reload the PPO value head saved next to an exported OpenWAM checkpoint.

    ``toolkits/openwam/export_ppo_checkpoint.py`` writes the architecture weights
    as a native OpenWAM checkpoint and keeps RLinf's ``value_head.*`` tensors in
    ``rlinf_value_head.pt``. OpenWAM's loader ignores that file, so a PPO run
    resumed from the export would otherwise start from a fresh critic. Returns
    True when a value head was loaded.
    """
    path = os.path.join(str(model_path), EXPORTED_VALUE_HEAD_FILE)
    if not os.path.isfile(path):
        return False
    state = torch.load(path, map_location="cpu", weights_only=True)
    prefix = "value_head."
    stripped = {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state.items()
    }
    expected = value_head.state_dict()
    if set(stripped) != set(expected):
        raise ValueError(
            f"{path} does not match the PPO value head: "
            f"file keys {sorted(stripped)}, expected {sorted(expected)}"
        )
    for key, value in stripped.items():
        if tuple(value.shape) != tuple(expected[key].shape):
            raise ValueError(
                f"{path} has {key} of shape {tuple(value.shape)}, "
                f"expected {tuple(expected[key].shape)}"
            )
    value_head.load_state_dict(
        {key: value.to(expected[key].dtype) for key, value in stripped.items()}
    )
    return True


class OpenWAMPolicy(nn.Module, BasePolicy):
    """RLinf adapter for OpenWAM inference and native SFT training."""

    def __init__(
        self,
        engine: Any,
        *,
        num_frames: int,
        height: int,
        width: int,
        denoise_steps: int,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        noise_std: float = 0.05,
        replay_text_capacity: int = 512,
        inference_horizon: int | None = None,
    ):
        super().__init__()
        self.engine = engine
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.denoise_steps = denoise_steps
        self.lambda_video = float(lambda_video)
        self.lambda_action = float(lambda_action)
        if noise_std <= 0:
            raise ValueError(f"OpenWAM PPO noise_std must be positive, got {noise_std}")
        self.noise_std = float(noise_std)
        self.replay_text_capacity = int(replay_text_capacity)
        if self.replay_text_capacity <= 0:
            raise ValueError("OpenWAM replay_text_capacity must be positive")
        # Receding-horizon eval: execute only the first ``inference_horizon``
        # actions of each generated chunk, as OpenWAM's deploy executor does.
        self.inference_horizon = (
            None if inference_horizon is None else int(inference_horizon)
        )
        if self.inference_horizon is not None and not (
            0 < self.inference_horizon <= max(1, num_frames - 1)
        ):
            raise ValueError(
                "OpenWAM inference_horizon must be in [1, num_frames - 1], got "
                f"{inference_horizon} for num_frames={num_frames}"
            )
        self.architecture = engine.architecture
        engine_cfg = getattr(engine, "cfg", None)
        dataloader_cfg = getattr(engine_cfg, "dataloader", None)
        self._multiview = bool(getattr(dataloader_cfg, "multiview", False))
        # RoboTwin-style readers wrap every instruction in a fixed sentence at
        # training time; LIBERO does not. Match the checkpoint's reader.
        self._prompt_template = _prompt_template_for_dataset(
            getattr(dataloader_cfg, "type", None)
        )
        # Checkpoints without an explicit layout (e.g. the LIBERO multiview
        # readers) use OpenWAM's three-slot canvas: head camera on top, two
        # wrist cameras below; missing slots stay black, as in the reader.
        layout = getattr(dataloader_cfg, "camera_layout", None)
        self._camera_layout = (
            list(layout)
            if layout is not None
            else ["head_camera", "left_camera", "right_camera"]
        )
        # PPO value features include pooled visual latents and proprioception.
        self.value_head = nn.Sequential(nn.Linear(8, 128), nn.SiLU(), nn.Linear(128, 1))

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        *,
        ckpt_name: str | None,
        device: str,
        torch_dtype: torch.dtype | None,
        num_frames: int,
        height: int,
        width: int,
        denoise_steps: int,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        encoder_model_path: str | None = None,
        noise_std: float = 0.05,
        replay_text_capacity: int = 512,
        inference_horizon: int | None = None,
        load_value_head: bool = True,
    ) -> "OpenWAMPolicy":
        """Build OpenWAM's checkpoint loader and joint inference engine."""
        from omegaconf import OmegaConf
        from openwam.deploy import JointInferenceEngine, load_from_checkpoint_dir

        # Checkpoints produced on the training host may retain an absolute
        # ``model.video_backbone.encoder.model_path``.  Alternate encoders
        # (DINOv3, Flux VAE, V-JEPA) are not always copied into the deploy
        # bundle, so allow the eval config to repoint that one dependency while
        # keeping the checkpoint itself immutable.
        with _checkpoint_with_encoder_override(
            model_path, encoder_model_path
        ) as load_dir:
            cfg, architecture = load_from_checkpoint_dir(
                load_dir, device=device, ckpt_name=ckpt_name
            )
        if OmegaConf.select(cfg, "inference", default=None) is None:
            cfg.inference = {}
        cfg.inference.num_frames = num_frames
        cfg.inference.denoise_steps = denoise_steps
        cfg.inference.height = height
        cfg.inference.width = width
        # Evaluation only consumes actions: skip the VAE decode of the generated
        # video. The engine reads this switch from cfg.inference.optimization,
        # not from the per-call condition dict.
        if OmegaConf.select(cfg, "inference.optimization", default=None) is None:
            cfg.inference.optimization = {}
        cfg.inference.optimization.decode_video = False
        freeze_names = OmegaConf.select(cfg, "model.freeze", default=[]) or []
        architecture.freeze_modules(list(freeze_names))
        training_cfg = OmegaConf.select(cfg, "training", default=OmegaConf.create({}))
        architecture.init_training_schedulers(1000)
        architecture.set_training_runtime(
            use_gradient_checkpointing=bool(
                OmegaConf.select(
                    training_cfg, "use_gradient_checkpointing", default=False
                )
            ),
            use_gradient_checkpointing_offload=bool(
                OmegaConf.select(
                    training_cfg, "use_gradient_checkpointing_offload", default=False
                )
            ),
            max_timestep_boundary=float(
                OmegaConf.select(training_cfg, "max_timestep_boundary", default=1.0)
            ),
            min_timestep_boundary=float(
                OmegaConf.select(training_cfg, "min_timestep_boundary", default=0.0)
            ),
        )
        model = cls(
            JointInferenceEngine(cfg=cfg, architecture=architecture),
            num_frames=num_frames,
            height=height,
            width=width,
            denoise_steps=denoise_steps,
            lambda_video=lambda_video,
            lambda_action=lambda_action,
            noise_std=noise_std,
            replay_text_capacity=replay_text_capacity,
            inference_horizon=inference_horizon,
        )
        model.to(
            device=device, dtype=torch_dtype or next(architecture.parameters()).dtype
        )
        if load_value_head and load_exported_value_head(model_path, model.value_head):
            logger.info(
                "Loaded PPO value head from %s/%s",
                model_path,
                EXPORTED_VALUE_HEAD_FILE,
            )
        return model

    def retarget_runtime_device(self, device: torch.device | str) -> None:
        """Point OpenWAM's cached input device at ``device`` without moving weights.

        OpenWAM's architecture and video backbones remember the device they
        prepare inputs on (text token ids, frame tensors, noise). The FSDP SFT
        path builds the policy on the CPU and lets FSDP move parameters and
        buffers, so that cache is retargeted here; ``set_dtype_device`` is not
        used because it would move the whole model.
        """
        device = torch.device(device)
        owners = [self.architecture]
        owners.extend((getattr(self.architecture, "backbones", None) or {}).values())
        for owner in owners:
            if hasattr(owner, "_device"):
                owner._device = device

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT or (
            forward_type == ForwardType.DEFAULT and "data" in kwargs
        ):
            return self.sft_forward(**kwargs)
        return self.default_forward(**kwargs)

    def default_forward(self, **kwargs):
        if "data" in kwargs:
            return self.sft_forward(**kwargs)
        return self.rl_forward(**kwargs)

    def rl_forward(
        self,
        forward_inputs,
        compute_logprobs=True,
        compute_values=True,
        compute_entropy=False,
        **kwargs,
    ):
        """Rescore the selected stochastic native denoising transition."""
        del kwargs
        chains = forward_inputs["chains"]
        predictions = []
        features = []
        for index in range(chains.shape[0]):
            chosen = int(forward_inputs["denoise_inds"][index, 0])
            action_t = chains[index : index + 1, chosen]
            if "native_schema" in forward_inputs:
                native = unpack_native_inputs(
                    {key: value[index] for key, value in forward_inputs.items()}
                )
            else:
                # Read trajectories collected by earlier versions of the adapter.
                native = {
                    key[len("native__") :]: value[index : index + 1]
                    for key, value in forward_inputs.items()
                    if key.startswith("native__")
                }
            native["latents"] = forward_inputs["video_latents"][index : index + 1]
            features.append(
                self._value_features(
                    {
                        "video_latents": native["latents"],
                        **{f"native__{key}": value for key, value in native.items()},
                    },
                    action_t,
                )
            )
            proprio = native.pop("proprio", None)
            action_timestep = (
                forward_inputs["action_timesteps"][index].reshape(1).to(action_t.dtype)
            )
            video_timestep = (
                forward_inputs["video_timesteps"][index].reshape(1).to(action_t.dtype)
            )
            _, prediction = self.architecture.forward(
                action_t,
                action_timestep,
                proprio=proprio,
                **native,
                timestep=video_timestep,
            )
            if prediction is None or prediction.shape != action_t.shape:
                raise ValueError(
                    "OpenWAM PPO requires an action flow prediction matching the action shape"
                )
            predictions.append(prediction)
        a_pred = torch.cat(predictions, dim=0)
        chosen = forward_inputs["denoise_inds"][:, 0].long()
        rows = torch.arange(chains.shape[0], device=chains.device)
        action_t, action_next = chains[rows, chosen], chains[rows, chosen + 1]
        sigma, sigma_next = (
            forward_inputs["sigma"].view(-1),
            forward_inputs["sigma_next"].view(-1),
        )
        mean = action_t + a_pred * (sigma_next - sigma).view(-1, 1, 1)
        std = forward_inputs["noise_std"].view(-1, 1, 1).clamp_min(1e-6)
        active = forward_inputs["active_action_indices"][0].long()
        if active.ndim != 1 or active.numel() == 0:
            raise ValueError(
                "OpenWAM PPO active_action_indices must be a non-empty vector"
            )
        if compute_logprobs:
            all_logprobs = (
                -0.5 * ((action_next - mean) / std).square()
                - torch.log(std)
                - 0.5 * np.log(2.0 * np.pi)
            )
            logprobs = all_logprobs.index_select(-1, active)
        else:
            logprobs = torch.zeros(
                action_next.shape[0],
                action_next.shape[1],
                active.numel(),
                device=action_next.device,
            )
        features = torch.cat(features, dim=0).to(
            next(self.value_head.parameters()).dtype
        )
        values = self.value_head(features).squeeze(-1)
        if not compute_values:
            values = torch.zeros(chains.shape[0], device=chains.device)
        entropy = (
            (torch.log(std) + 0.5 * np.log(2.0 * np.pi * np.e)).expand_as(logprobs)
            if compute_entropy
            else None
        )
        if self.inference_horizon is not None:
            # Only the executed prefix of the chunk is the RL action; the
            # unexecuted tail is regenerated at the next step.
            logprobs = logprobs[:, : self.inference_horizon]
            if entropy is not None:
                entropy = entropy[:, : self.inference_horizon]
        return {
            "logprobs": logprobs.float(),
            "values": values.float(),
            "entropy": entropy,
        }

    @staticmethod
    def _value_features(inputs, action):
        def stats(x):
            # Native proprio is a (D,) vector per observation; treat it as one sample.
            x = x.float()
            if x.ndim < 2:
                x = x.unsqueeze(0)
            x = x.flatten(1)
            return x.mean(1), x.std(1, unbiased=False)

        vm, vs = stats(inputs["video_latents"])
        context = inputs.get("native__context", inputs["video_latents"])
        cm, cs = stats(context)
        proprio = inputs.get("native__proprio", action)
        pm, ps = stats(proprio)
        am, astd = stats(action)
        return torch.stack((vm, vs, cm, cs, pm, ps, am, astd), dim=-1)

    def sft_forward(self, data: Any = None, **kwargs) -> dict[str, torch.Tensor]:
        """Compute OpenWAM's native joint video/action SFT loss."""
        if data is None:
            data = kwargs.get("batch")
        if data is None:
            raise ValueError(
                "OpenWAM sft_forward requires `data` from the SFT dataloader."
            )
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, (list, tuple)):
            raise TypeError(
                f"OpenWAM SFT data must be a sample list, got {type(data)!r}"
            )
        inputs = self.architecture.prepare_inputs(list(data))
        result = self.architecture.compute_loss(
            **inputs,
            lambda_video=float(kwargs.get("lambda_video", self.lambda_video)),
            lambda_action=float(kwargs.get("lambda_action", self.lambda_action)),
        )
        return {
            "loss": result["loss"],
            "loss_video": result.get("loss_video", result["loss"].detach()),
            "loss_action": result.get("loss_action", result["loss"].detach()),
        }

    def _format_prompt(self, prompt: Any) -> str:
        """Apply the checkpoint reader's instruction template, if it has one."""
        text = "" if prompt is None else str(prompt)
        if self._prompt_template is None:
            return text
        return self._prompt_template + text

    def predict_action_batch(
        self, env_obs: dict[str, Any], mode: str = "eval", **kwargs
    ):
        """Generate one action chunk per observation in ``env_obs``.

        OpenWAM's deploy engine generates one condition at a time, so this
        runs ``len(env_obs)`` sequential generations; evaluation cost grows
        linearly with ``total_num_envs``.
        """
        if mode == "train":
            return self._predict_rl_batch(env_obs)
        if mode != "eval":
            raise ValueError(f"Unknown OpenWAM predict mode: {mode!r}")
        batch_size = _infer_batch_size(env_obs)
        actions, results = [], []
        for index in range(batch_size):
            main_image = _to_pil(
                _batch_value(env_obs, ("main_images", "image", "images"), index)
            )
            wrist_image = _batch_value(
                env_obs, ("wrist_images", "wrist_image"), index, None
            )
            image = _compose_observation_image(
                main_image,
                wrist_image,
                multiview=self._multiview,
                camera_layout=self._camera_layout,
                height=self.height,
                width=self.width,
            )
            condition = {
                "prompt": self._format_prompt(
                    _batch_value(
                        env_obs,
                        ("task_descriptions", "task_description", "language"),
                        index,
                        "",
                    )
                ),
                "first_frame_image": [image],
                "proprio": _observation_proprio(env_obs, index),
                "num_frames": self.num_frames,
                "height": self.height,
                "width": self.width,
                "denoise_steps": self.denoise_steps,
            }
            result = self.engine.generate(condition)
            actions.append(np.asarray(result["actions"]))
            results.append(result)
        stacked = torch.as_tensor(np.stack(actions), dtype=torch.float32)
        if self.inference_horizon is not None:
            stacked = stacked[:, : self.inference_horizon]
        return stacked, {"results": results}

    @torch.no_grad()
    def _predict_rl_batch(self, env_obs):
        """Collect a native joint-flow chain for PPO."""
        # OpenWAM architecture variants share the joint-flow forward contract.
        # Tri-system additionally consumes VLM inputs; those are prepared below
        # and kept in the replay record. Fail early for future architectures
        # that do not expose the scheduler/backbone interface required here.
        for name in ("video_backbone", "action_scheduler", "video_scheduler"):
            if not hasattr(self.architecture, name):
                raise NotImplementedError(
                    f"OpenWAM PPO requires architecture.{name}; "
                    f"unsupported architecture {self.architecture.__class__.__name__}."
                )
        batch_size = _infer_batch_size(env_obs)
        records, outputs = [], []
        steps = max(2, self.denoise_steps)
        parameter = next(self.architecture.parameters())
        device, dtype = parameter.device, parameter.dtype
        action_dim = int(self.architecture.action_dim)
        for index in range(batch_size):
            main = _to_pil(
                _batch_value(env_obs, ("main_images", "image", "images"), index)
            )
            wrist = _batch_value(env_obs, ("wrist_images", "wrist_image"), index, None)
            image = _compose_observation_image(
                main,
                wrist,
                multiview=self._multiview,
                camera_layout=self._camera_layout,
                height=self.height,
                width=self.width,
            )
            proprio = _observation_proprio(env_obs, index)
            action_shift = (
                getattr(self.architecture.action_backbone, "shift_action", None) or 5.0
            )
            video_shift = (
                getattr(self.architecture.video_backbone, "shift_video", None)
                or action_shift
            )
            native = self.architecture.video_backbone.preprocess_input_for_inference(
                prompt=self._format_prompt(
                    _batch_value(
                        env_obs,
                        ("task_descriptions", "task_description", "language"),
                        index,
                        "",
                    )
                ),
                first_frame_image=[image],
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                seed=42,
                num_inference_steps=steps,
                shift=float(video_shift),
                tiled=True,
                cfg_scale=1.0,
                cfg_merge=False,
            )
            # The native tri-system engine caches the frozen VLM once per
            # observation; retain the same features for differentiable replay.
            vlm_backbone = getattr(self.architecture, "vlm_backbone", None)
            if vlm_backbone is not None:
                if any(param.requires_grad for param in vlm_backbone.parameters()):
                    raise NotImplementedError(
                        "OpenWAM tri-system PPO requires a frozen VLM backbone"
                    )
                prompt = self._format_prompt(
                    prompt=_batch_value(
                        env_obs,
                        ("task_descriptions", "task_description", "language"),
                        index,
                        "",
                    )
                )
                vlm_inputs = vlm_backbone.prepare_vlm_inputs([prompt], [image])
                native["vlm_hidden"] = vlm_backbone.extract_features(
                    vlm_inputs
                ).detach()
                native["vlm_attention_mask"] = vlm_inputs.get("attention_mask")
                if native["vlm_attention_mask"] is not None:
                    native["vlm_attention_mask"] = native["vlm_attention_mask"].to(
                        device
                    )
            if getattr(self.architecture, "uses_proprioception", False):
                native["proprio"] = self.architecture.normalize_deploy_proprio(
                    proprio
                ).to(device=device, dtype=dtype)
            video = _restore_clean_prefix(
                native["latents"], native.get("first_frame_latents")
            )
            native["latents"] = video
            action = torch.randn(
                1, max(1, self.num_frames - 1), action_dim, device=device, dtype=dtype
            )
            from openwam.deploy.denoise_schedule import make_schedule

            schedule = make_schedule(
                "sync",
                self.architecture.video_scheduler,
                self.architecture.action_scheduler,
                num_steps=steps,
                shift=float(action_shift),
                shift_video=float(video_shift),
            )
            chosen = int(torch.randint(0, len(schedule) - 1, ()).item())
            chains = [action]
            selected_video = selected_sigma = selected_next = selected_std = None
            selected_action_timestep = selected_video_timestep = None
            for step, ((tv, ta), (tv_next, ta_next)) in enumerate(
                zip(schedule[:-1], schedule[1:])
            ):
                sigma = torch.tensor(
                    [ta / self.architecture.action_scheduler.num_train_timesteps],
                    device=device,
                    dtype=dtype,
                )
                sigma_next = torch.tensor(
                    [ta_next / self.architecture.action_scheduler.num_train_timesteps],
                    device=device,
                    dtype=dtype,
                )
                video_sigma = torch.tensor(
                    [tv / self.architecture.video_scheduler.num_train_timesteps],
                    device=device,
                    dtype=dtype,
                )
                video_sigma_next = torch.tensor(
                    [tv_next / self.architecture.video_scheduler.num_train_timesteps],
                    device=device,
                    dtype=dtype,
                )
                native_pipeline = dict(native)
                proprio_native = native_pipeline.pop("proprio", None)
                vp, ap = self.architecture.forward(
                    action,
                    torch.tensor([ta], device=device, dtype=dtype),
                    proprio=proprio_native,
                    **native_pipeline,
                    timestep=torch.tensor([tv], device=device, dtype=dtype),
                )
                if step == chosen:
                    selected_video = native["latents"].detach().clone()
                video = _restore_clean_prefix(
                    video + vp * (video_sigma_next - video_sigma),
                    native.get("first_frame_latents"),
                )
                native["latents"] = video
                mean = action + ap * (sigma_next - sigma).view(1, 1, 1)
                noise_std = (
                    torch.sqrt((sigma - sigma_next).clamp_min(1e-6)) * self.noise_std
                )
                if step == chosen:
                    selected_sigma, selected_next, selected_std = (
                        sigma.detach(),
                        sigma_next.detach(),
                        noise_std.detach(),
                    )
                    selected_action_timestep = torch.tensor(
                        [ta], device=device, dtype=torch.float32
                    )
                    selected_video_timestep = torch.tensor(
                        [tv], device=device, dtype=torch.float32
                    )
                    action = mean + torch.randn_like(action) * noise_std
                else:
                    action = mean
                chains.append(action)
            flat = {
                "chains": torch.stack(chains, dim=1).squeeze(0).contiguous(),
                "denoise_inds": torch.full(
                    (steps,), chosen, device=device, dtype=torch.long
                ),
                "video_latents": selected_video.squeeze(0).contiguous(),
                "sigma": selected_sigma,
                "sigma_next": selected_next,
                "noise_std": selected_std,
                "video_timesteps": selected_video_timestep,
                "action_timesteps": selected_action_timestep,
                "active_action_indices": torch.as_tensor(
                    getattr(
                        getattr(self.architecture, "normalizer", None),
                        "_dst_index",
                        list(range(action_dim)),
                    ),
                    device=device,
                    dtype=torch.long,
                ),
                "model_action": action.squeeze(0).reshape(-1).float(),
            }
            physical = action.squeeze(0).float().cpu().numpy()
            normalizer = getattr(self.architecture, "normalizer", None)
            if normalizer is not None:
                physical = normalizer.unnormalize(physical)
            flat["action"] = (
                torch.as_tensor(physical, device=device).reshape(-1).float()
            )
            flat.update(
                pack_native_inputs(native, text_capacity=self.replay_text_capacity)
            )
            records.append(flat)
            outputs.append(
                torch.as_tensor(physical, device=device, dtype=torch.float32)
            )
        forward_inputs = _stack_flat_records(records)
        # Rollout sampling is performed one observation at a time.  Keep the
        # behavior policy score in that same batch shape: the Wan attention
        # path can produce slightly different bf16 results for batch=1 versus
        # batch>1, and actor training commonly uses micro_batch_size=1.
        # Scoring the stacked batch here would make PPO ratios non-unit even
        # before an optimizer update.
        per_record_scores = []
        for index in range(batch_size):
            record_inputs = {
                key: value[index : index + 1] for key, value in forward_inputs.items()
            }
            per_record_scores.append(
                self.rl_forward(record_inputs, compute_values=True)
            )
        executed = torch.stack(outputs)
        if self.inference_horizon is not None:
            executed = executed[:, : self.inference_horizon]
        return executed, {
            "prev_logprobs": torch.cat(
                [score["logprobs"].detach() for score in per_record_scores], dim=0
            ),
            "prev_values": torch.cat(
                [score["values"].detach().unsqueeze(-1) for score in per_record_scores],
                dim=0,
            ),
            "forward_inputs": forward_inputs,
        }


@contextmanager
def _checkpoint_with_encoder_override(
    model_path: str, encoder_model_path: str | None
) -> Iterator[str]:
    """Yield a checkpoint directory with deploy-time encoder fixes applied.

    OpenWAM checkpoints can retain training-host paths and legacy encoder names
    (``vjepa2_1``, ``flux_vae``, ``wan_vae``). A temporary staging directory
    rewrites only those config fields without editing the checkpoint or source
    checkout. All other files are symlinked, so this adds no model-storage cost.
    """
    checkpoint_dir = Path(model_path).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"OpenWAM checkpoint directory not found: {checkpoint_dir}"
        )

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    encoder_cfg = OmegaConf.select(cfg, "model.video_backbone.encoder", default=None)
    encoder_name = (
        str(getattr(encoder_cfg, "name", "")) if encoder_cfg is not None else ""
    )
    name_aliases = {
        "vjepa2_1": "vjepa21",
        "flux_vae": "flux2_vae",
        "wan_vae": "wan22_vae",
    }
    canonical_name = name_aliases.get(encoder_name, encoder_name)
    needs_staging = encoder_model_path is not None or canonical_name != encoder_name
    if not needs_staging:
        yield model_path
        return
    if encoder_cfg is None:
        raise ValueError(
            "OpenWAM encoder override/alias normalization requires "
            "model.video_backbone.encoder in the checkpoint config"
        )
    if encoder_model_path is not None:
        encoder_dir = Path(encoder_model_path).expanduser().resolve()
        if not encoder_dir.is_dir():
            raise FileNotFoundError(
                f"OpenWAM encoder_model_path is not a directory: {encoder_dir}"
            )
        encoder_cfg.model_path = str(encoder_dir)
    if canonical_name != encoder_name:
        encoder_cfg.name = canonical_name

    with tempfile.TemporaryDirectory(prefix="rlinf-openwam-") as tmp:
        staged = Path(tmp) / checkpoint_dir.name
        staged.mkdir()
        for entry in checkpoint_dir.iterdir():
            if entry.name != "config.yaml":
                os.symlink(
                    entry, staged / entry.name, target_is_directory=entry.is_dir()
                )
        OmegaConf.save(cfg, staged / "config.yaml")
        yield str(staged)


def _infer_batch_size(obs: dict[str, Any]) -> int:
    for key in ("states", "state", "proprio", "main_images", "image", "images"):
        if key in obs:
            value = obs[key]
            if isinstance(value, (torch.Tensor, np.ndarray)):
                return int(value.shape[0]) if value.ndim > 1 else 1
            if isinstance(value, (list, tuple)):
                return len(value)
    return 1


def _batch_value(
    obs: dict[str, Any], keys: tuple[str, ...], index: int, default: Any = None
) -> Any:
    value = next((obs[key] for key in keys if key in obs), default)
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim > 1:
        return value[index]
    if isinstance(value, (list, tuple)) and value and len(value) > index:
        return value[index]
    return value


def _to_pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (list, tuple)):
        value = value[0]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.dtype != np.uint8:
        array = np.clip(array * 255 if array.max() <= 1.0 else array, 0, 255).astype(
            np.uint8
        )
    return Image.fromarray(array)


def _axis_angle_to_rotation_6d(value: Any) -> np.ndarray:
    """Convert a LIBERO axis-angle state to OpenWAM's 6-D representation."""
    from scipy.spatial.transform import Rotation

    matrix = Rotation.from_rotvec(
        np.asarray(value, dtype=np.float64).reshape(3)
    ).as_matrix()
    return np.concatenate([matrix[:, 0], matrix[:, 1]]).astype(np.float32)


def _libero_state_to_eef10(value: Any) -> np.ndarray | None:
    """Convert RLinf's 8-D LIBERO state to OpenWAM's achieved EEF10 state."""
    if value is None:
        return None
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    if state.shape[0] == 10:
        return state
    if state.shape[0] != 8:
        raise ValueError(
            "OpenWAM LIBERO rollout expects an 8-D state "
            f"(xyz, axis-angle, gripper qpos), got {state.shape[0]}"
        )
    gripper_width = float(state[6] - state[7])
    gripper_open_scale = np.clip(2.0 * gripper_width / 0.08 - 1.0, -1.0, 1.0)
    return np.concatenate(
        [state[:3], _axis_angle_to_rotation_6d(state[3:6]), [gripper_open_scale]]
    ).astype(np.float32)


ROBOTWIN_PROMPT_PREFIX = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: "
)
_TEMPLATED_DATASET_TYPES = ("robotwin", "robodojo", "ebench")


def _prompt_template_for_dataset(dataset_type: Any) -> str | None:
    """Return the training-time instruction prefix of an OpenWAM dataset reader."""
    if dataset_type is None or str(dataset_type) not in _TEMPLATED_DATASET_TYPES:
        return None
    try:
        from openwam.dataloader.transforms.multiview import (
            format_prompt_for_inference,
        )

        return format_prompt_for_inference("")
    except ImportError:
        return ROBOTWIN_PROMPT_PREFIX


def _wrist_frames(value: Any) -> list[Image.Image]:
    """Split a wrist observation into per-camera PIL frames, in layout order.

    RLinf environments hand back either one wrist image (``[H, W, 3]``) or a
    stack of them (``[n, H, W, 3]``; RoboTwin stacks left then right).
    """
    if value is None:
        return []
    if isinstance(value, Image.Image):
        return [value]
    if isinstance(value, (list, tuple)):
        return [_to_pil(item) for item in value if item is not None]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        return [_to_pil(frame) for frame in array]
    return [_to_pil(array)]


def _compose_observation_image(
    main_image: Image.Image,
    wrist_image: Any,
    *,
    multiview: bool,
    camera_layout: list[str],
    height: int,
    width: int,
) -> Image.Image:
    """Match OpenWAM's single-view or three-camera training layout.

    Single-view checkpoints see the head camera center-cropped and resized to
    the training canvas, as ``openwam.deploy.obs_preprocess`` does. Multiview
    checkpoints get the head camera in slot 0 and the wrist cameras in the
    following slots of ``camera_layout``; missing cameras stay black, exactly
    like the dataset reader pads them.
    """
    from openwam.dataloader.transforms.multiview import (
        assemble_multiview_layout,
        crop_and_resize,
    )

    if not multiview:
        if main_image.size == (width, height):
            return main_image
        return crop_and_resize(main_image, height, width)

    frames = {camera_layout[0]: main_image}
    for slot, frame in zip(camera_layout[1:], _wrist_frames(wrist_image)):
        frames[slot] = frame
    return assemble_multiview_layout(
        frames,
        camera_layout=camera_layout,
        out_h=height,
        out_w=width,
    )


def _observation_proprio(env_obs: dict[str, Any], index: int) -> np.ndarray | None:
    """Use explicit native physical units, or convert LIBERO's achieved pose."""
    value = _batch_value(env_obs, ("native_proprio",), index, None)
    if value is not None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError(
                "native_proprio must be a finite vector in checkpoint physical units"
            )
        return value
    return _libero_state_to_eef10(
        _batch_value(env_obs, ("states", "state", "proprio"), index, None)
    )


def _restore_clean_prefix(
    video: torch.Tensor, reference: torch.Tensor | None
) -> torch.Tensor:
    if reference is not None:
        video = video.clone()
        video[:, :, : reference.shape[2]] = reference
    return video


def _stack_flat_records(
    records: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack the flat, fixed-shape tensors consumed by RLinf trajectories."""
    if not records:
        raise ValueError("OpenWAM RL sampler produced no records")
    keys = set(records[0])
    if any(set(record) != keys for record in records[1:]):
        raise ValueError(
            "OpenWAM RL records have inconsistent native conditioning keys"
        )
    return {
        key: torch.stack([record[key] for record in records], dim=0).contiguous()
        for key in keys
    }
