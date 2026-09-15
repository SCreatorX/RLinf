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

"""Check the OpenWAM PPO bridge against one real checkpoint without an environment.

Loads the checkpoint through ``OpenWAMPolicy.from_checkpoint``, samples two
observations in rollout mode, pushes the recorded ``forward_inputs`` through the
same split/stack/flatten/shuffle transport RLinf applies to trajectories, rescores
them with the actor path and backpropagates. A healthy bridge reports
``ratio_per_sample`` of exactly 1.0 (identical weights), finite log-probabilities
and finite gradients on the action backbone. Results are written as JSON.

Example::

    PYTHONPATH=$PWD python toolkits/openwam/check_ppo_replay.py \
        /path/to/OpenWAM_Study_Checkpoints/architecture_study/robotwin_tri_system_joint_self_attention \
        --out /tmp/tri_system.json

Study checkpoints trained on Robotwin carry a 20-D state; the check feeds zeros
of the checkpoint's ``proprio_dim`` through ``native_proprio``. ``--replay-rows 1``
keeps a 14B backbone within one GPU; ``--no-normalizer`` is a diagnostic for
checkpoints whose ``normalization_stats.npy`` does not match ``state_dim``.
"""

import argparse
import json
import sys
import time
import traceback

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("--out", required=True)
p.add_argument("--denoise-steps", type=int, default=2)
p.add_argument(
    "--replay-rows",
    type=int,
    default=4,
    help="rows of the emulated (T*B) batch to replay; lower to fit one GPU",
)
p.add_argument(
    "--no-normalizer",
    action="store_true",
    help="smoke-only: bypass a normalizer whose stats do not match state_dim",
)
args = p.parse_args()
res = {"ckpt": args.ckpt.rstrip("/").split("/")[-1], "ok": False}
t0 = time.time()
try:
    from rlinf.models.embodiment.openwam.openwam_policy import OpenWAMPolicy

    policy = OpenWAMPolicy.from_checkpoint(
        args.ckpt,
        ckpt_name=None,
        device="cuda",
        torch_dtype=None,
        num_frames=33,
        height=384,
        width=320,
        denoise_steps=args.denoise_steps,
    )
    arch = policy.architecture
    res["arch"] = arch.__class__.__name__
    res["video_backbone"] = arch.video_backbone.__class__.__name__
    res["proprio_dim"] = int(getattr(arch, "proprio_dim", 0))
    # Checkpoints trained with dataloader.unify_action scatter a physical state
    # (e.g. 20-D) into a wider unified vector (e.g. 80-D); feed the physical width.
    normalizer = getattr(arch, "normalizer", None)
    state_index = getattr(normalizer, "_state_dst_index", None)
    physical_dim = len(state_index) if state_index is not None else res["proprio_dim"]
    res["physical_state_dim"] = int(physical_dim)
    if args.no_normalizer:
        res["normalizer_bypassed"] = True
        arch.normalizer = None
    res["action_dim"] = int(arch.action_dim)
    res["load_s"] = round(time.time() - t0, 1)
    vlm = getattr(arch, "vlm_backbone", None)
    if vlm is not None:
        res["vlm_requires_grad_after_load"] = any(
            q.requires_grad for q in vlm.parameters()
        )
        vlm.requires_grad_(False)
    n_train = sum(q.numel() for q in arch.parameters() if q.requires_grad)
    res["trainable_params_M"] = round(n_train / 1e6, 1)

    rng = np.random.default_rng(0)
    obs = {
        "images": rng.integers(0, 255, size=(2, 384, 320, 3), dtype=np.uint8),
        "native_proprio": np.zeros((2, physical_dim or 20), dtype=np.float32),
        "task_descriptions": [
            "pick up the bottle",
            "place the red block into the box on the left side of the table",
        ],
    }
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    actions, extra = policy.predict_action_batch(obs, mode="train")
    res["rollout_s"] = round(time.time() - t1, 1)
    res["actions_shape"] = list(actions.shape)
    fi = extra["forward_inputs"]
    res["forward_input_keys"] = len(fi)
    res["all_tensors"] = all(isinstance(v, torch.Tensor) for v in fi.values())
    res["transport_MB"] = round(
        sum(v.numel() * v.element_size() for v in fi.values()) / 2**20, 1
    )
    # RLinf transport: split per env on B, stack on T, flatten T/B, shuffle.
    split = [
        {k: torch.split(v, [1, 1], dim=0)[i] for k, v in fi.items()} for i in range(2)
    ]
    per_step = {k: torch.cat([split[0][k], split[1][k]], dim=0) for k in fi}  # (B, ...)
    perm = torch.tensor([3, 0, 1, 2])
    batch = {
        k: torch.stack([v, v], dim=0).flatten(0, 1)[perm] for k, v in per_step.items()
    }  # (T*B, ...) shuffled
    orig_index = torch.tensor([0, 1, 0, 1])[perm]
    rows = args.replay_rows
    batch = {k: v[:rows] for k, v in batch.items()}
    prev = extra["prev_logprobs"][orig_index][:rows]
    res["replay_rows"] = rows

    policy.train()
    t2 = time.time()
    out = policy.rl_forward(
        batch, compute_logprobs=True, compute_values=True, compute_entropy=True
    )
    res["replay_s"] = round(time.time() - t2, 1)
    new = out["logprobs"]
    diff = (new - prev).abs()
    res["logprob_max_abs_diff"] = float(diff.max())
    res["logprob_mean_abs_diff"] = float(diff.mean())
    res["ratio_per_sample"] = [
        round(float(x), 4) for x in torch.exp((new - prev).sum(dim=(1, 2)))
    ]
    res["values"] = [round(float(x), 4) for x in out["values"]]
    res["logprobs_finite"] = bool(torch.isfinite(new).all())
    loss = -new.mean() + out["values"].mean()
    t3 = time.time()
    loss.backward()
    res["backward_s"] = round(time.time() - t3, 1)
    grads = [(n, q.grad) for n, q in arch.named_parameters() if q.grad is not None]
    res["params_with_grad"] = len(grads)
    res["grads_finite"] = all(torch.isfinite(g).all().item() for _, g in grads)

    def gnorm(prefix):
        return float(
            torch.sqrt(
                sum(g.float().square().sum() for n, g in grads if n.startswith(prefix))
                or torch.tensor(0.0)
            )
        )

    res["grad_norm_action_backbone"] = gnorm("action_backbone")
    res["grad_norm_video_backbone"] = gnorm("video_backbone")
    res["grad_norm_value_head"] = float(
        torch.sqrt(
            sum(
                q.grad.float().square().sum()
                for q in policy.value_head.parameters()
                if q.grad is not None
            )
        )
    )
    res["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    res["ok"] = bool(
        res["logprobs_finite"]
        and res["grads_finite"]
        and res["params_with_grad"] > 0
        and res["logprob_max_abs_diff"] < 1e-2
    )
except Exception as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {e}"
    res["traceback"] = traceback.format_exc()[-3000:]
res["total_s"] = round(time.time() - t0, 1)
with open(args.out, "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "traceback"}))
sys.exit(0 if res["ok"] else 1)
