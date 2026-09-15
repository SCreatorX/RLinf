# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0

"""Preserve native OpenWAM conditioning in RLinf's flat tensor trajectory."""

import json
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

_SCHEMA_BYTES = 16384
# These image objects have already been encoded into y/CLIP/VACE tensors.
_PREPROCESS_ONLY = {
    "input_image",
    "end_image",
    "input_video",
    "input_audio",
    "control_video",
    "reference_image",
    "vace_video",
    "vace_video_mask",
    "vace_reference_image",
}
_TEXT_INPUTS = {
    "context",
    "context_mask",
    "und_mask",
    "und_kv",
    "vlm_hidden",
    "vlm_attention_mask",
}


def pack_native_inputs(
    inputs: dict[str, Any], *, text_capacity: int
) -> dict[str, torch.Tensor]:
    """Pack one observation, retaining tensor shapes, nested caches and scalars.

    Text axes are padded only for transport, to a fixed capacity across rollout
    chunks. Unpacking removes padding before the architecture sees the tensors.
    The caller stacks the returned tensors on RLinf's sample axis.
    """
    tensors: dict[str, torch.Tensor] = {}

    def encode(value: Any, root: str) -> list:
        if isinstance(value, torch.Tensor):
            key = f"native_tensor_{len(tensors):03d}"
            shape = list(value.shape)
            value = value.detach()
            if root in _TEXT_INPUTS and value.ndim >= 2:
                if value.shape[1] > text_capacity:
                    raise ValueError(
                        f"OpenWAM {root} text length {value.shape[1]} exceeds replay "
                        f"capacity {text_capacity}. Increase openwam.replay_text_capacity."
                    )
                padding = [0, 0] * value.ndim
                padding[2 * (value.ndim - 2) + 1] = text_capacity - value.shape[1]
                value = F.pad(value, padding)
            tensors[key] = value.contiguous()
            return ["tensor", key, shape]
        if isinstance(value, Mapping):
            return [
                "dict",
                [[key, encode(item, root)] for key, item in sorted(value.items())],
            ]
        if isinstance(value, (tuple, list)):
            return [
                "tuple" if isinstance(value, tuple) else "list",
                [encode(item, root) for item in value],
            ]
        if value is None or isinstance(value, (str, bool, int, float)):
            return ["scalar", value]
        raise TypeError(
            f"Unsupported OpenWAM replay conditioning {root}: {type(value)!r}"
        )

    schema = [
        "dict",
        [
            [key, encode(value, key)]
            for key, value in sorted(inputs.items())
            if key not in _PREPROCESS_ONLY
            and key not in {"latents", "noise", "input_latents"}
        ],
    ]
    payload = json.dumps(schema, separators=(",", ":"), allow_nan=False).encode()
    if len(payload) > _SCHEMA_BYTES:
        raise ValueError(f"OpenWAM replay schema exceeds {_SCHEMA_BYTES} bytes")
    device = inputs["latents"].device
    tensors["native_schema"] = torch.tensor(
        list(payload) + [0] * (_SCHEMA_BYTES - len(payload)),
        dtype=torch.uint8,
        device=device,
    )
    return tensors


def unpack_native_inputs(record: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Restore one sample after RLinf has removed its leading sample axis."""
    payload = bytes(record["native_schema"].detach().cpu().tolist()).rstrip(b"\x00")
    schema = json.loads(payload)

    def decode(node: list) -> Any:
        kind = node[0]
        if kind == "tensor":
            value = record[node[1]]
            return value[tuple(slice(0, size) for size in node[2])].contiguous()
        if kind == "dict":
            return {key: decode(value) for key, value in node[1]}
        if kind == "tuple":
            return tuple(decode(value) for value in node[1])
        if kind == "list":
            return [decode(value) for value in node[1]]
        if kind == "scalar":
            return node[1]
        raise ValueError(f"Unknown OpenWAM replay schema node: {kind!r}")

    return decode(schema)
