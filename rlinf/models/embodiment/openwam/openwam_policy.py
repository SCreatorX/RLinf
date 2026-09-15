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


class OpenWAMPolicy(nn.Module, BasePolicy):
    """RLinf adapter for OpenWAM inference and native SFT training."""

    def __init__(self, engine: Any, *, num_frames: int, height: int, width: int, denoise_steps: int,
                 lambda_video: float = 1.0, lambda_action: float = 1.0,
                 noise_std: float = 0.05):
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
        self.architecture = engine.architecture
        engine_cfg = getattr(engine, "cfg", None)
        dataloader_cfg = getattr(engine_cfg, "dataloader", None)
        self._multiview = bool(getattr(dataloader_cfg, "multiview", False))
        layout = getattr(dataloader_cfg, "camera_layout", None)
        self._camera_layout = list(layout) if layout is not None else [
            "head_camera", "left_camera", "right_camera"
        ]
        # PPO value features include pooled visual latents and proprioception.
        self.value_head = nn.Sequential(nn.Linear(8, 128), nn.SiLU(), nn.Linear(128, 1))

    @classmethod
    def from_checkpoint(cls, model_path: str, *, ckpt_name: str | None, device: str,
                        torch_dtype: torch.dtype | None, num_frames: int, height: int,
                        width: int, denoise_steps: int, lambda_video: float = 1.0,
                        lambda_action: float = 1.0,
                        encoder_model_path: str | None = None,
                        noise_std: float = 0.05) -> "OpenWAMPolicy":
        """Build OpenWAM's checkpoint loader and joint inference engine."""
        from omegaconf import OmegaConf
        from openwam.deploy import JointInferenceEngine, load_from_checkpoint_dir

        # Checkpoints produced on the training host may retain an absolute
        # ``model.video_backbone.encoder.model_path``.  Alternate encoders
        # (DINOv3, Flux VAE, V-JEPA) are not always copied into the deploy
        # bundle, so allow the eval config to repoint that one dependency while
        # keeping the checkpoint itself immutable.
        with _checkpoint_with_encoder_override(model_path, encoder_model_path) as load_dir:
            cfg, architecture = load_from_checkpoint_dir(
                load_dir, device=device, ckpt_name=ckpt_name
            )
        if OmegaConf.select(cfg, "inference", default=None) is None:
            cfg.inference = {}
        cfg.inference.num_frames = num_frames
        cfg.inference.denoise_steps = denoise_steps
        cfg.inference.height = height
        cfg.inference.width = width
        freeze_names = OmegaConf.select(cfg, "model.freeze", default=[]) or []
        architecture.freeze_modules(list(freeze_names))
        training_cfg = OmegaConf.select(cfg, "training", default=OmegaConf.create({}))
        architecture.init_training_schedulers(1000)
        architecture.set_training_runtime(
            use_gradient_checkpointing=bool(OmegaConf.select(training_cfg, "use_gradient_checkpointing", default=False)),
            use_gradient_checkpointing_offload=bool(OmegaConf.select(training_cfg, "use_gradient_checkpointing_offload", default=False)),
            max_timestep_boundary=float(OmegaConf.select(training_cfg, "max_timestep_boundary", default=1.0)),
            min_timestep_boundary=float(OmegaConf.select(training_cfg, "min_timestep_boundary", default=0.0)),
        )
        model = cls(JointInferenceEngine(cfg=cfg, architecture=architecture),
                    num_frames=num_frames, height=height, width=width,
                    denoise_steps=denoise_steps, lambda_video=lambda_video,
                    lambda_action=lambda_action, noise_std=noise_std)
        model.to(device=device, dtype=torch_dtype or next(architecture.parameters()).dtype)
        return model

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT or (forward_type == ForwardType.DEFAULT and "data" in kwargs):
            return self.sft_forward(**kwargs)
        return self.default_forward(**kwargs)

    def default_forward(self, **kwargs):
        if "data" in kwargs:
            return self.sft_forward(**kwargs)
        return self.rl_forward(**kwargs)

    def rl_forward(self, forward_inputs, compute_logprobs=True, compute_values=True,
                   compute_entropy=False, **kwargs):
        """Rescore the selected stochastic native denoising transition."""
        del kwargs
        chains = forward_inputs["chains"]
        chosen = forward_inputs["denoise_inds"][:, 0].long()
        rows = torch.arange(chains.shape[0], device=chains.device)
        action_t, action_next = chains[rows, chosen], chains[rows, chosen + 1]
        native = {k[len("native__"):]: v for k, v in forward_inputs.items() if k.startswith("native__")}
        native["latents"] = forward_inputs["video_latents"]
        proprio = native.pop("proprio", None)
        _, a_pred = self.architecture.forward(action_t, forward_inputs["action_timesteps"].view(-1),
                                               proprio=proprio, **native,
                                               timestep=forward_inputs["video_timesteps"].view(-1))
        sigma, sigma_next = forward_inputs["sigma"].view(-1), forward_inputs["sigma_next"].view(-1)
        mean = action_t + a_pred * (sigma_next - sigma).view(-1, 1, 1)
        std = forward_inputs["noise_std"].view(-1, 1, 1).clamp_min(1e-6)
        active = forward_inputs["active_action_indices"][0].long()
        if active.ndim != 1 or active.numel() == 0:
            raise ValueError("OpenWAM PPO active_action_indices must be a non-empty vector")
        if compute_logprobs:
            all_logprobs = (-0.5 * ((action_next - mean) / std).square()
                            - torch.log(std) - 0.5 * np.log(2.0 * np.pi))
            logprobs = all_logprobs.index_select(-1, active)
        else:
            logprobs = torch.zeros(action_next.shape[0], action_next.shape[1], active.numel(), device=action_next.device)
        features = self._value_features(forward_inputs, action_t).to(next(self.value_head.parameters()).dtype)
        values = self.value_head(features).squeeze(-1)
        if not compute_values:
            values = torch.zeros(chains.shape[0], device=chains.device)
        entropy = (torch.log(std) + 0.5 * np.log(2.0 * np.pi * np.e)).expand_as(logprobs) if compute_entropy else None
        return {"logprobs": logprobs.float(), "values": values.float(), "entropy": entropy}

    @staticmethod
    def _value_features(inputs, action):
        def stats(x):
            x = x.float().flatten(1)
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
            raise ValueError("OpenWAM sft_forward requires `data` from the SFT dataloader.")
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, (list, tuple)):
            raise TypeError(f"OpenWAM SFT data must be a sample list, got {type(data)!r}")
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

    def predict_action_batch(self, env_obs: dict[str, Any], mode: str = "eval", **kwargs):
        """Generate one action chunk per observation in ``env_obs``."""
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
                _to_pil(wrist_image) if wrist_image is not None else None,
                multiview=self._multiview,
                camera_layout=self._camera_layout,
                height=self.height,
                width=self.width,
            )
            condition = {
                "prompt": _batch_value(env_obs, ("task_descriptions", "task_description", "language"), index, ""),
                "first_frame_image": [image],
                "proprio": _libero_state_to_eef10(
                    _batch_value(env_obs, ("states", "state", "proprio"), index, None)
                ),
                "num_frames": self.num_frames, "height": self.height, "width": self.width,
                "denoise_steps": self.denoise_steps, "decode_video": False,
            }
            result = self.engine.generate(condition)
            actions.append(np.asarray(result["actions"]))
            results.append(result)
        return torch.as_tensor(np.stack(actions), dtype=torch.float32), {"results": results}

    @torch.no_grad()
    def _predict_rl_batch(self, env_obs):
        """Collect a native joint-flow chain for PPO."""
        if self.architecture.__class__.__name__ != "DualSystemSelfAttnArchitecture":
            raise NotImplementedError(
                "OpenWAM PPO rollout currently supports only the validated "
                "dual_system_self_attn (Wan22) architecture."
            )
        batch_size = _infer_batch_size(env_obs)
        records, outputs = [], []
        steps = max(2, self.denoise_steps)
        parameter = next(self.architecture.parameters())
        device, dtype = parameter.device, parameter.dtype
        action_dim = int(self.architecture.action_dim)
        for index in range(batch_size):
            main = _to_pil(_batch_value(env_obs, ("main_images", "image", "images"), index))
            wrist = _batch_value(env_obs, ("wrist_images", "wrist_image"), index, None)
            image = _compose_observation_image(main, _to_pil(wrist) if wrist is not None else None,
                                                multiview=self._multiview, camera_layout=self._camera_layout,
                                                height=self.height, width=self.width)
            proprio = _libero_state_to_eef10(_batch_value(env_obs, ("states", "state", "proprio"), index, None))
            action_shift = getattr(self.architecture.action_backbone, "shift_action", None) or 5.0
            video_shift = getattr(self.architecture.video_backbone, "shift_video", None) or action_shift
            native = self.architecture.video_backbone.preprocess_input_for_inference(
                prompt=_batch_value(env_obs, ("task_descriptions", "task_description", "language"), index, ""),
                first_frame_image=[image], num_frames=self.num_frames, height=self.height, width=self.width,
                seed=42, num_inference_steps=steps, shift=float(video_shift), tiled=True,
                cfg_scale=1.0, cfg_merge=False)
            native = {k: v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) else v for k, v in native.items()}
            if getattr(self.architecture, "uses_proprioception", False):
                native["proprio"] = self.architecture.normalize_deploy_proprio(proprio).to(device=device, dtype=dtype)
            video = native["latents"]
            action = torch.randn(1, max(1, self.num_frames - 1), action_dim, device=device, dtype=dtype)
            from openwam.deploy.denoise_schedule import make_schedule
            schedule = make_schedule("sync", self.architecture.video_scheduler,
                                     self.architecture.action_scheduler, num_steps=steps,
                                     shift=float(action_shift), shift_video=float(video_shift))
            chosen = int(torch.randint(0, len(schedule) - 1, ()).item())
            chains = [action]
            selected_video = selected_sigma = selected_next = selected_std = None
            for step, ((tv, ta), (tv_next, ta_next)) in enumerate(zip(schedule[:-1], schedule[1:])):
                sigma = torch.tensor([ta / self.architecture.action_scheduler.num_train_timesteps], device=device, dtype=dtype)
                sigma_next = torch.tensor([ta_next / self.architecture.action_scheduler.num_train_timesteps], device=device, dtype=dtype)
                video_sigma = torch.tensor([tv / self.architecture.video_scheduler.num_train_timesteps], device=device, dtype=dtype)
                video_sigma_next = torch.tensor([tv_next / self.architecture.video_scheduler.num_train_timesteps], device=device, dtype=dtype)
                native_pipeline = dict(native)
                proprio_native = native_pipeline.pop("proprio", None)
                vp, ap = self.architecture.forward(action, torch.tensor([ta], device=device, dtype=dtype),
                                                   proprio=proprio_native, **native_pipeline,
                                                   timestep=torch.tensor([tv], device=device, dtype=dtype))
                if step == chosen:
                    selected_video = native["latents"].detach().clone()
                video = video + vp * (video_sigma_next - video_sigma)
                native["latents"] = video
                mean = action + ap * (sigma_next - sigma).view(1, 1, 1)
                noise_std = torch.sqrt((sigma - sigma_next).clamp_min(1e-6)) * self.noise_std
                if step == chosen:
                    selected_sigma, selected_next, selected_std = sigma.detach(), sigma_next.detach(), noise_std.detach()
                    action = mean + torch.randn_like(action) * noise_std
                else:
                    action = mean
                chains.append(action)
            flat = {"chains": torch.stack(chains, dim=1).squeeze(0).contiguous(),
                    "denoise_inds": torch.full((steps,), chosen, device=device, dtype=torch.long),
                    "video_latents": selected_video.squeeze(0).contiguous(), "sigma": selected_sigma,
                    "sigma_next": selected_next, "noise_std": selected_std,
                    "video_timesteps": selected_sigma * 1000.0, "action_timesteps": selected_sigma * 1000.0,
                    "active_action_indices": torch.as_tensor(getattr(getattr(self.architecture, "normalizer", None), "_dst_index", list(range(action_dim))), device=device, dtype=torch.long),
                    "model_action": action.squeeze(0).reshape(-1).float()}
            physical = action.squeeze(0).float().cpu().numpy()
            normalizer = getattr(self.architecture, "normalizer", None)
            if normalizer is not None:
                physical = normalizer.unnormalize(physical)
            flat["action"] = torch.as_tensor(physical, device=device).reshape(-1).float()
            for key, value in native.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if key == "proprio" and value.ndim == 1:
                    value = value.unsqueeze(0)
                if value.ndim > 0 and value.shape[0] == 1:
                    flat[f"native__{key}"] = value.contiguous()
            records.append(flat)
            outputs.append(torch.as_tensor(physical, device=device, dtype=torch.float32))
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
            per_record_scores.append(self.rl_forward(record_inputs, compute_values=True))
        return torch.stack(outputs), {
            "prev_logprobs": torch.cat(
                [score["logprobs"].detach() for score in per_record_scores], dim=0
            ),
            "prev_values": torch.cat(
                [score["values"].detach().unsqueeze(-1) for score in per_record_scores], dim=0
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
        raise FileNotFoundError(f"OpenWAM checkpoint directory not found: {checkpoint_dir}")

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    encoder_cfg = OmegaConf.select(cfg, "model.video_backbone.encoder", default=None)
    encoder_name = str(getattr(encoder_cfg, "name", "")) if encoder_cfg is not None else ""
    name_aliases = {"vjepa2_1": "vjepa21", "flux_vae": "flux2_vae", "wan_vae": "wan22_vae"}
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
            raise FileNotFoundError(f"OpenWAM encoder_model_path is not a directory: {encoder_dir}")
        encoder_cfg.model_path = str(encoder_dir)
    if canonical_name != encoder_name:
        encoder_cfg.name = canonical_name

    with tempfile.TemporaryDirectory(prefix="rlinf-openwam-") as tmp:
        staged = Path(tmp) / checkpoint_dir.name
        staged.mkdir()
        for entry in checkpoint_dir.iterdir():
            if entry.name != "config.yaml":
                os.symlink(entry, staged / entry.name, target_is_directory=entry.is_dir())
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


def _batch_value(obs: dict[str, Any], keys: tuple[str, ...], index: int, default: Any = None) -> Any:
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
        array = np.clip(array * 255 if array.max() <= 1.0 else array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _axis_angle_to_rotation_6d(value: Any) -> np.ndarray:
    """Convert a LIBERO axis-angle state to OpenWAM's 6-D representation."""
    from scipy.spatial.transform import Rotation

    matrix = Rotation.from_rotvec(np.asarray(value, dtype=np.float64).reshape(3)).as_matrix()
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


def _compose_observation_image(
    main_image: Image.Image,
    wrist_image: Image.Image | None,
    *,
    multiview: bool,
    camera_layout: list[str],
    height: int,
    width: int,
) -> Image.Image:
    """Match OpenWAM's single-view or three-camera training layout."""
    if not multiview:
        return main_image
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout

    frames = {camera_layout[0]: main_image}
    if wrist_image is not None and len(camera_layout) > 1:
        frames[camera_layout[1]] = wrist_image
    return assemble_multiview_layout(
        frames,
        camera_layout=camera_layout,
        out_h=height,
        out_w=width,
    )


def _stack_flat_records(records: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-observation tensors while preserving the rollout flat contract."""
    if not records:
        raise ValueError("OpenWAM RL sampler produced no records")
    keys = set(records[0])
    if any(set(record) != keys for record in records[1:]):
        raise ValueError("OpenWAM RL records have inconsistent native conditioning keys")
    result = {}
    for key in keys:
        values = [record[key] for record in records]
        if key.startswith("native__") and values[0].ndim > 0 and values[0].shape[0] == 1:
            result[key] = torch.cat(values, dim=0).contiguous()
        else:
            result[key] = torch.stack(values, dim=0).contiguous()
    return result


def _stack_flat_records(records: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-observation tensors while preserving the rollout contract."""
    if not records:
        raise ValueError("OpenWAM RL sampler produced no records")
    keys = set(records[0])
    if any(set(record) != keys for record in records[1:]):
        raise ValueError("OpenWAM RL records have inconsistent native conditioning keys")
    result = {}
    for key in keys:
        values = [record[key] for record in records]
        if key.startswith("native__") and values[0].ndim > 0 and values[0].shape[0] == 1:
            result[key] = torch.cat(values, dim=0).contiguous()
        else:
            result[key] = torch.stack(values, dim=0).contiguous()
    return result
