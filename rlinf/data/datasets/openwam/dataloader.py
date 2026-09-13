# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Build an RLinf SFT loader from OpenWAM's native dataset readers."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import ListConfig, OmegaConf


def build_openwam_sft_dataloader(cfg: Any, world_size: int, rank: int, data_paths: Any,
                                 eval_dataset: bool = False) -> tuple[Any, dict[str, Any]]:
    """Construct a distributed DataLoader yielding lists of native samples."""
    if isinstance(data_paths, (list, tuple, ListConfig)):
        if len(data_paths) != 1:
            raise ValueError("OpenWAM SFT currently accepts exactly one dataset path.")
        data_paths = data_paths[0]
    if data_paths is None:
        raise ValueError("OpenWAM SFT requires data.train_data_paths.")

    model_cfg = cfg.actor.model
    model_path = str(model_cfg.model_path)
    config_path = OmegaConf.select(model_cfg, "openwam_config_path", default=None)
    if config_path is None:
        config_path = f"{model_path}/config.yaml"
    native_cfg = OmegaConf.load(str(config_path))
    if not hasattr(native_cfg, "dataloader"):
        raise ValueError(f"OpenWAM config has no dataloader section: {config_path}")
    native_dl = native_cfg.dataloader.copy()
    native_dl.dataset_dir = str(data_paths)
    native_dl.split = "val" if eval_dataset else "train"
    overrides = OmegaConf.select(cfg, "data.openwam", default=None)
    if overrides is not None:
        native_dl = OmegaConf.merge(native_dl, overrides)

    from openwam.dataloader.registry import build_dataset

    dataset = build_dataset(native_dl, split=str(native_dl.split))
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=int(world_size), rank=int(rank),
        shuffle=not eval_dataset, drop_last=True,
        seed=int(OmegaConf.select(cfg, "actor.seed", default=0)),
    )
    batch_size = int(cfg.actor.get("eval_batch_size", cfg.actor.micro_batch_size)) if eval_dataset else int(cfg.actor.micro_batch_size)
    num_workers = int(OmegaConf.select(cfg, "data.num_workers", default=0))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
        collate_fn=list, pin_memory=True, drop_last=True,
    )
    return loader, {"dataset_type": str(native_dl.type), "dataset_dir": str(data_paths), "num_samples": len(dataset)}
