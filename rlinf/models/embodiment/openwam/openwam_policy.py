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

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class OpenWAMPolicy(nn.Module, BasePolicy):
    """RLinf adapter for OpenWAM inference and native SFT training."""

    def __init__(self, engine: Any, *, num_frames: int, height: int, width: int, denoise_steps: int,
                 lambda_video: float = 1.0, lambda_action: float = 1.0):
        super().__init__()
        self.engine = engine
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.denoise_steps = denoise_steps
        self.lambda_video = float(lambda_video)
        self.lambda_action = float(lambda_action)
        self.architecture = engine.architecture

    @classmethod
    def from_checkpoint(cls, model_path: str, *, ckpt_name: str | None, device: str,
                        torch_dtype: torch.dtype | None, num_frames: int, height: int,
                        width: int, denoise_steps: int, lambda_video: float = 1.0,
                        lambda_action: float = 1.0) -> "OpenWAMPolicy":
        """Build OpenWAM's checkpoint loader and joint inference engine."""
        from omegaconf import OmegaConf
        from openwam.deploy import JointInferenceEngine, load_from_checkpoint_dir

        cfg, architecture = load_from_checkpoint_dir(model_path, device=device, ckpt_name=ckpt_name)
        if OmegaConf.select(cfg, "inference", default=None) is None:
            cfg.inference = {}
        cfg.inference.num_frames = num_frames
        cfg.inference.denoise_steps = denoise_steps
        cfg.inference.height = height
        cfg.inference.width = width
        training_cfg = OmegaConf.select(cfg, "training", default={})
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
                    lambda_action=lambda_action)
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)
        return model

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT or (forward_type == ForwardType.DEFAULT and "data" in kwargs):
            return self.sft_forward(**kwargs)
        return self.default_forward(**kwargs)

    def default_forward(self, **kwargs):
        if "data" in kwargs:
            return self.sft_forward(**kwargs)
        raise NotImplementedError(
            "OpenWAM default_forward requires SFT data; use predict_action_batch for rollout evaluation."
        )

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
        if mode != "eval":
            raise NotImplementedError("OpenWAM rollout training mode is not implemented yet.")
        batch_size = _infer_batch_size(env_obs)
        actions, results = [], []
        for index in range(batch_size):
            condition = {
                "prompt": _batch_value(env_obs, ("task_descriptions", "task_description", "language"), index, ""),
                "first_frame_image": [_to_pil(_batch_value(env_obs, ("main_images", "image", "images"), index))],
                "proprio": _batch_value(env_obs, ("states", "state", "proprio"), index),
                "num_frames": self.num_frames, "height": self.height, "width": self.width,
                "denoise_steps": self.denoise_steps, "decode_video": False,
            }
            result = self.engine.generate(condition)
            actions.append(np.asarray(result["actions"]))
            results.append(result)
        return torch.as_tensor(np.stack(actions), dtype=torch.float32), {"results": results}


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
