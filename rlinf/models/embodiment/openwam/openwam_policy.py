"""RLinf policy wrapper around OpenWAM's native deployment engine."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from rlinf.models.embodiment.base_policy import BasePolicy


class OpenWAMPolicy(nn.Module, BasePolicy):
    """Batch adapter for OpenWAM checkpoint-first inference.

    This first milestone supports evaluation only. PPO log-probabilities and
    the IDM cached action path are added after the LIBERO observation/action
    contract is validated.
    """

    def __init__(self, engine: Any, *, num_frames: int, height: int, width: int, denoise_steps: int):
        super().__init__()
        self.engine = engine
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.denoise_steps = denoise_steps
        self.architecture = engine.architecture

    @classmethod
    def from_checkpoint(cls, model_path: str, *, ckpt_name: str | None, device: str,
                        torch_dtype: torch.dtype | None, num_frames: int, height: int,
                        width: int, denoise_steps: int) -> "OpenWAMPolicy":
        """Build OpenWAM's checkpoint loader and joint inference engine."""
        from openwam.deploy import JointInferenceEngine, load_from_checkpoint_dir
        from omegaconf import OmegaConf

        cfg, architecture = load_from_checkpoint_dir(model_path, device=device, ckpt_name=ckpt_name)
        if OmegaConf.select(cfg, "inference", default=None) is None:
            cfg.inference = {}
        cfg.inference.num_frames = num_frames
        cfg.inference.denoise_steps = denoise_steps
        cfg.inference.height = height
        cfg.inference.width = width
        model = cls(JointInferenceEngine(cfg=cfg, architecture=architecture),
                    num_frames=num_frames, height=height, width=width,
                    denoise_steps=denoise_steps)
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)
        return model

    def default_forward(self, **kwargs):
        raise NotImplementedError("OpenWAM Phase 1 supports evaluation only; RL default_forward is Phase 3.")

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
