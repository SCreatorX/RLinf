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
    openwam_cfg = cfg.get("openwam", {}) or {}
    model = OpenWAMPolicy.from_checkpoint(
        model_path=str(model_path),
        ckpt_name=cfg.get("ckpt_name"),
        device=str(cfg.get("device", "cuda")),
        torch_dtype=torch_dtype,
        num_frames=int(cfg.get("num_frames", 49)),
        height=int(cfg.get("height", 384)),
        width=int(cfg.get("width", 320)),
        denoise_steps=int(cfg.get("denoise_steps", 10)),
        lambda_video=float(cfg.get("lambda_video", 1.0)),
        lambda_action=float(cfg.get("lambda_action", 1.0)),
        noise_std=float((cfg.get("openwam", {}) or {}).get("noise_std", 0.05)),
        replay_text_capacity=int(
            (cfg.get("openwam", {}) or {}).get("replay_text_capacity", 512)
        ),
        inference_horizon=(cfg.get("openwam", {}) or {}).get("inference_horizon"),
        encoder_model_path=(
            str(cfg.get("encoder_model_path"))
            if cfg.get("encoder_model_path") is not None
            else None
        ),
    )
    # The PPO value head always exists so rollout can score values and exported
    # checkpoints stay uniform, but it only trains under RL. SFT and eval keep
    # it frozen: otherwise FSDP's optimizer builds a critic parameter group and
    # demands ``actor.optim.value_lr`` from recipes that never mention a critic.
    if not bool(openwam_cfg.get("rl_enabled", False)):
        for parameter in model.value_head.parameters():
            parameter.requires_grad_(False)
    return model


__all__ = ["OpenWAMPolicy", "get_model"]
