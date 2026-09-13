"""OpenWAM model loading for RLinf embodied evaluation."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.openwam.openwam_policy import OpenWAMPolicy


def get_model(cfg: DictConfig, torch_dtype: torch.dtype | None = None) -> OpenWAMPolicy:
    """Load an OpenWAM policy from a self-contained checkpoint directory."""
    model_path = cfg.get("model_path")
    if not model_path:
        raise ValueError("OpenWAM requires actor.model.model_path to be set.")
    return OpenWAMPolicy.from_checkpoint(
        model_path=str(model_path),
        ckpt_name=cfg.get("ckpt_name"),
        device=str(cfg.get("device", "cuda")),
        torch_dtype=torch_dtype,
        num_frames=int(cfg.get("num_frames", 49)),
        height=int(cfg.get("height", 384)),
        width=int(cfg.get("width", 320)),
        denoise_steps=int(cfg.get("denoise_steps", 10)),
    )


__all__ = ["OpenWAMPolicy", "get_model"]
