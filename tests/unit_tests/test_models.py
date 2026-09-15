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

"""Model registration, embeddings, and the reward-model helpers."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.config import SupportedModel
from rlinf.envs.action_utils import (
    _openwam_absolute_eef10_to_libero7,
    _openwam_eef10_to_libero7,
)
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models import get_model, register_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTTokenTransformer,
)
from rlinf.models.embodiment.openwam.openwam_policy import (
    OpenWAMPolicy,
    _batch_value,
    _checkpoint_with_encoder_override,
    _infer_batch_size,
    _libero_state_to_eef10,
    _to_pil,
)
from rlinf.models.embodiment.openwam.replay import (
    pack_native_inputs,
    unpack_native_inputs,
)
from rlinf.utils.env_helpers import HistoryManager
from rlinf.utils.env_helpers.delay_sampler import (
    ConstantDelaySampler,
    DelaySampler,
    ExponentialDelaySampler,
    GaussianDelaySampler,
    UniformDelaySampler,
)
from toolkits.openwam.export_ppo_checkpoint import (
    export_checkpoint,
    infer_step,
)


class _DummyModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class _DummyFSDPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _DummyBlock()
        self.head = torch.nn.Linear(4, 2)
        self.head._fsdp_wrap_name = "custom_head"


@pytest.fixture
def openwam_recipe(monkeypatch):
    import hydra

    import rlinf.config as config_module

    placement = SimpleNamespace(
        get_world_size=lambda component: {"actor": 2, "env": 1, "rollout": 1}[component]
    )
    monkeypatch.setattr(config_module, "Cluster", lambda: object())
    monkeypatch.setattr(
        config_module, "HybridComponentPlacement", lambda cfg, cluster: placement
    )
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"

    def load(name, overrides=None):
        with hydra.initialize_config_dir(
            version_base="1.1", config_dir=str(config_dir)
        ):
            return hydra.compose(config_name=name, overrides=overrides or [])

    return load


@pytest.mark.parametrize(
    "name",
    [
        "libero_spatial_ppo_openwam",
        "libero_spatial_ppo_openwam_smoke",
        "libero_spatial_ppo_openwam_long",
    ],
)
def test_openwam_ppo_recipes_validate(openwam_recipe, name):
    from rlinf.config import validate_embodied_cfg

    cfg = openwam_recipe(name)
    assert validate_embodied_cfg(cfg) is cfg


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            ["actor.global_batch_size=2048", "actor.micro_batch_size=128"],
            r"5120 action-chunk samples.*global_batch_size \(2048\)",
        ),
        (["actor.global_batch_size=0"], "batch sizes must be positive"),
        (["actor.micro_batch_size=0"], "batch sizes must be positive"),
        (["actor.micro_batch_size=3"], r"micro_batch_size \* actor_world_size"),
    ],
)
def test_openwam_ppo_rejects_invalid_batch_before_rollout(
    openwam_recipe, overrides, message
):
    from rlinf.config import validate_embodied_cfg

    cfg = openwam_recipe("libero_spatial_ppo_openwam", overrides)
    with pytest.raises(AssertionError, match=message):
        validate_embodied_cfg(cfg)


def test_openwam_observation_adapter_smoke():
    observations = {
        "states": np.zeros((2, 7), dtype=np.float32),
        "main_images": np.zeros((2, 16, 16, 3), dtype=np.uint8),
        "task_descriptions": ["pick", "place"],
    }

    assert _infer_batch_size(observations) == 2
    assert _batch_value(observations, ("states",), 1).shape == (7,)
    assert _to_pil(_batch_value(observations, ("main_images",), 0)).size == (16, 16)


def test_openwam_libero_state_adapter_smoke():
    state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, 0.04], dtype=np.float32)
    eef10 = _libero_state_to_eef10(state)
    assert eef10.shape == (10,)
    np.testing.assert_allclose(eef10[:3], state[:3])
    np.testing.assert_allclose(eef10[3:9], [1, 0, 0, 0, 1, 0])
    assert eef10[9] == -1.0


def test_openwam_libero_absolute_action_adapter_smoke():
    pose = np.array([[0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 1]], dtype=np.float32)
    target = pose.copy()
    target[0, 0] += 0.025
    converted = _openwam_absolute_eef10_to_libero7(target, pose)
    np.testing.assert_allclose(converted[0, :3], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(converted[0, 3:6], 0.0)
    assert converted[0, 6] == -1.0


def test_openwam_libero_action_adapter_smoke():
    action = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 1]], dtype=np.float32)
    converted = _openwam_eef10_to_libero7(action)
    assert converted.shape == (1, 7)
    np.testing.assert_allclose(converted[0, :6], 0.0)
    assert converted[0, 6] == -1.0


def test_openwam_libero_action_adapter_rejects_nonfinite_values():
    pose = np.array([[0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 1]], dtype=np.float32)
    invalid = pose.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _openwam_absolute_eef10_to_libero7(invalid, pose)
    with pytest.raises(ValueError, match="non-finite"):
        _openwam_eef10_to_libero7(invalid)


def test_openwam_encoder_path_override_stages_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_step_1.safetensors").write_bytes(b"weights")
    (checkpoint / "normalization_stats.npy").write_bytes(b"stats")
    (checkpoint / "config.yaml").write_text(
        "model:\n  video_backbone:\n    encoder:\n      name: vjepa2_1\n      model_path: /old/host/path\n",
        encoding="utf-8",
    )
    encoder = tmp_path / "encoder"
    encoder.mkdir()

    with _checkpoint_with_encoder_override(
        str(checkpoint), str(encoder)
    ) as staged_path:
        staged = Path(staged_path)
        cfg = OmegaConf.load(staged / "config.yaml")
        assert cfg.model.video_backbone.encoder.model_path == str(encoder.resolve())
        assert (staged / "checkpoint_step_1.safetensors").is_symlink()
        assert (staged / "normalization_stats.npy").is_symlink()

    assert not Path(staged_path).exists()


def test_openwam_replay_survives_flat_trajectory_transport():
    records = []
    for length in (2, 4):
        native = {
            "latents": torch.zeros(1, 1, 2, 2, 2),
            "context": torch.arange(length * 3).reshape(1, length, 3).float(),
            "seq_lens": torch.tensor([length], dtype=torch.long),
            "und_kv": [(torch.ones(1, length, 2, 3), torch.zeros(1, length, 2, 3))],
            "vision_positions": torch.arange(12).reshape(3, 1, 4),
            "num_clean_prefix_frames": 1,
            "cfg_merge": False,
            "und_mask": None,
        }
        records.append(pack_native_inputs(native, text_capacity=8))
    assert all(
        isinstance(value, torch.Tensor)
        for record in records
        for value in record.values()
    )
    # Rollout workers split flat tensors on B; trajectories stack on T, then
    # actor training flattens T/B and shuffles samples.
    batch = {
        key: torch.stack([record[key] for record in records]) for key in records[0]
    }
    time_batch = {key: torch.stack([value, value]) for key, value in batch.items()}
    shuffled = {
        key: value.flatten(0, 1)[torch.tensor([1, 0, 3, 2])]
        for key, value in time_batch.items()
    }
    restored = unpack_native_inputs({key: value[0] for key, value in shuffled.items()})
    assert restored["context"].shape == (1, 4, 3)
    assert isinstance(restored["und_kv"], list)
    assert isinstance(restored["und_kv"][0], tuple)
    assert restored["und_kv"][0][0].shape == (1, 4, 2, 3)
    assert restored["vision_positions"].shape == (3, 1, 4)
    assert restored["seq_lens"].dtype == torch.long
    assert restored["num_clean_prefix_frames"] == 1
    assert restored["cfg_merge"] is False
    assert restored["und_mask"] is None
    with pytest.raises(ValueError, match="exceeds replay capacity"):
        pack_native_inputs(native, text_capacity=2)


@pytest.mark.parametrize("tri_system", [False, True])
@pytest.mark.parametrize("horizon", [None, 1])
def test_openwam_architecture_rollout_replay_and_backward(
    monkeypatch, tri_system, horizon
):
    # The external OpenWAM scheduler returns real (unrounded) schedule values.
    schedule = ModuleType("openwam.deploy.denoise_schedule")
    schedule.make_schedule = lambda *args, **kwargs: [
        (1000.0, 1000.0),
        (909.09, 909.09),
        (0.0, 0.0),
    ]
    monkeypatch.setitem(sys.modules, "openwam.deploy.denoise_schedule", schedule)

    class VideoBackbone(torch.nn.Module):
        def preprocess_input_for_inference(self, **kwargs):
            length = len(kwargs["prompt"])
            return {
                "latents": torch.zeros(1, 1, 2, 2, 2),
                "first_frame_latents": torch.ones(1, 1, 1, 2, 2),
                "context": torch.ones(1, length, 3),
                "und_kv": [(torch.ones(1, length, 1, 3), torch.zeros(1, length, 1, 3))],
                "vision_positions": torch.arange(12).reshape(3, 1, 4),
                "num_clean_prefix_frames": 1,
                "seq_lens": torch.tensor([length]),
            }

    class FrozenVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=False)
            self.calls = 0

        def prepare_vlm_inputs(self, prompts, images):
            return {"attention_mask": torch.ones(1, len(prompts[0]), dtype=torch.long)}

        def extract_features(self, inputs):
            self.calls += 1
            return torch.ones(1, inputs["attention_mask"].shape[1], 4)

    class OtherArchitecture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
            self.video_backbone = VideoBackbone()
            self.action_backbone = SimpleNamespace(shift_action=5.0)
            self.action_scheduler = SimpleNamespace(num_train_timesteps=1000)
            self.video_scheduler = SimpleNamespace(num_train_timesteps=1000)
            self.action_dim = 20
            self.uses_proprioception = True
            if tri_system:
                self.vlm_backbone = FrozenVLM()

        def normalize_deploy_proprio(self, value):
            assert value.shape == (20,)
            return torch.from_numpy(value)

        def forward(self, action, action_timestep, *, latents, proprio, **native):
            assert native["num_clean_prefix_frames"] == 1
            assert torch.equal(latents[:, :, :1], native["first_frame_latents"])
            assert native["vision_positions"].shape == (3, 1, 4)
            assert isinstance(native["und_kv"][0], tuple)
            length = native["context"].shape[1]
            assert native["und_kv"][0][0].shape[1] == length
            if tri_system:
                assert native["vlm_hidden"].shape[1] == length
                assert native["vlm_attention_mask"].shape == (1, length)
            return torch.ones_like(latents), action * self.weight

    architecture = OtherArchitecture()
    policy = OpenWAMPolicy(
        SimpleNamespace(architecture=architecture),
        num_frames=3,
        height=8,
        width=8,
        denoise_steps=2,
        replay_text_capacity=8,
        inference_horizon=horizon,
    )
    obs = {
        "images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "native_proprio": np.zeros((2, 20), dtype=np.float32),
        "task_descriptions": ["ab", "abcd"],
    }
    actions, extra = policy.predict_action_batch(obs, mode="train")
    executed = 2 if horizon is None else horizon
    assert actions.shape == (2, executed, 20)
    assert extra["prev_logprobs"].shape[1] == executed
    # The replay inputs keep the full chain so the actor can rescore it.
    assert extra["forward_inputs"]["chains"].shape[2] == 2
    assert all(
        isinstance(value, torch.Tensor) for value in extra["forward_inputs"].values()
    )
    policy.train()
    replay = policy(forward_inputs=extra["forward_inputs"], compute_entropy=True)
    torch.testing.assert_close(
        replay["logprobs"], extra["prev_logprobs"], rtol=0, atol=0
    )
    loss = -replay["logprobs"].mean() + replay["values"].square().mean()
    loss.backward()
    assert torch.isfinite(architecture.weight.grad)
    assert architecture.weight.grad.abs() > 0
    if tri_system:
        assert architecture.vlm_backbone.calls == 2
        architecture.vlm_backbone.weight.requires_grad_(True)
        with pytest.raises(NotImplementedError, match="frozen VLM"):
            policy.predict_action_batch(obs, mode="train")


def test_openwam_eval_inference_horizon_truncates_chunks():
    """Eval executes the first ``inference_horizon`` actions of a 32-step chunk."""

    class _Engine:
        architecture = SimpleNamespace(action_dim=10)

        def generate(self, condition):
            return {"actions": np.tile(np.arange(32, dtype=np.float32)[:, None], 10)}

    obs = {
        "images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "task_descriptions": ["a", "b"],
    }
    full = OpenWAMPolicy(_Engine(), num_frames=33, height=8, width=8, denoise_steps=2)
    actions, _ = full.predict_action_batch(obs, mode="eval")
    assert actions.shape == (2, 32, 10)

    policy = OpenWAMPolicy(
        _Engine(),
        num_frames=33,
        height=8,
        width=8,
        denoise_steps=2,
        inference_horizon=10,
    )
    actions, _ = policy.predict_action_batch(obs, mode="eval")
    assert actions.shape == (2, 10, 10)
    assert torch.equal(actions[0, :, 0], torch.arange(10, dtype=torch.float32))

    with pytest.raises(ValueError, match="inference_horizon"):
        OpenWAMPolicy(
            _Engine(),
            num_frames=33,
            height=8,
            width=8,
            denoise_steps=2,
            inference_horizon=33,
        )


def test_openwam_sft_forward_delegates_native_loss():
    class _Architecture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen = None

        def prepare_inputs(self, data):
            self.seen = data
            return {"marker": torch.tensor(1.0)}

        def compute_loss(self, **kwargs):
            assert kwargs["lambda_video"] == 0.25
            assert kwargs["lambda_action"] == 0.75
            return {
                "loss": self.weight * 2,
                "loss_video": self.weight.detach(),
                "loss_action": self.weight.detach() * 3,
            }

    architecture = _Architecture()
    engine = type("_Engine", (), {"architecture": architecture})()
    policy = OpenWAMPolicy(
        engine,
        num_frames=33,
        height=384,
        width=320,
        denoise_steps=10,
        lambda_video=0.25,
        lambda_action=0.75,
    )
    sample = {"video": [], "prompt": "pick"}
    output = policy(forward_type=ForwardType.SFT, data=[sample])
    assert output["loss"].requires_grad
    assert architecture.seen == [sample]
    assert output["loss_action"].item() == 3


def test_custom_model_registration_smoke():
    model_type = f"custom_model_smoke_{int(time.time() * 1000)}"
    received = {"torch_dtype": None}

    def _builder(cfg, torch_dtype):
        received["torch_dtype"] = torch_dtype
        return _DummyModel()

    register_model(model_type, _builder, category="embodied")

    supported_model = SupportedModel(model_type)
    assert supported_model.value == model_type

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    model = get_model(cfg)

    assert isinstance(model, _DummyModel)
    assert received["torch_dtype"] == torch.float32


def test_custom_model_registration_with_fsdp_wrap_policy():
    model_type = f"custom_model_fsdp_{int(time.time() * 1000)}"

    def _builder(cfg, torch_dtype):
        return _DummyFSDPModel()

    register_model(
        model_type,
        _builder,
        category="embodied",
    )

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    fsdp_cfg = OmegaConf.create(
        {
            "wrap_policy": {
                "transformer_layer_cls_to_wrap": ["_DummyBlock"],
                "module_classes_to_wrap": ["_DummyBlock"],
                "no_split_names": ["custom_head"],
            },
            "use_orig_params": True,
        }
    )
    model = get_model(cfg)
    wrap_policy = get_fsdp_wrap_policy(
        module=model,
        config=fsdp_cfg,
        is_lora=False,
        model_type=model_type,
    )

    assert wrap_policy is not None
    assert wrap_policy(module=model.block, recurse=False, nonwrapped_numel=0)
    assert wrap_policy(module=model.head, recurse=False, nonwrapped_numel=0)


def _make_model(*, prefix_seq_len: int = 5) -> RLTTokenTransformer:
    torch.manual_seed(0)
    return RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        prefix_seq_len=prefix_seq_len,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
    )


def test_decoder_causal_mask_blocks_future_teacher_targets():
    model = _make_model()
    model.eval()
    rl_tokens = torch.randn(1, 1, model.embed_dim)
    targets = torch.randn(1, model.prefix_seq_len, model.input_dim)

    changed_targets = targets.clone()
    changed_targets[:, 2:] += 100.0

    original_output = model.decode(rl_tokens, targets)
    changed_output = model.decode(rl_tokens, changed_targets)

    # target[2:] enters decoder positions 3+, so positions 0..2 must not
    # change when causal attention prevents access to future positions.
    torch.testing.assert_close(
        original_output[:, :3],
        changed_output[:, :3],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(original_output[:, 3:], changed_output[:, 3:])


def test_loss_masks_trailing_padding():
    model = _make_model(prefix_seq_len=4)
    model.eval()
    prefix_embs = torch.randn(2, 4, model.input_dim)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    loss, _ = model.loss(prefix_embs, mask)
    reconstructed, _ = model.reconstruct(prefix_embs, mask)
    valid = mask.unsqueeze(-1).to(dtype=torch.float32)
    expected_loss = (
        torch.square(reconstructed.float() - prefix_embs.float()) * valid
    ).sum() / (valid.sum() * model.input_dim)
    torch.testing.assert_close(loss, expected_loss)

    changed_padding = prefix_embs.clone()
    changed_padding[~mask] += 1000.0
    changed_loss, _ = model.loss(changed_padding, mask)
    torch.testing.assert_close(loss, changed_loss, rtol=1e-5, atol=1e-5)


def test_reconstruct_output_shape_matches_prefix_embeddings():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(3, 4, model.input_dim)

    reconstructed, _ = model.reconstruct(prefix_embs)

    assert reconstructed.shape == prefix_embs.shape


def test_reconstruct_detaches_targets_but_trains_encoder_and_decoder():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(2, 4, model.input_dim, requires_grad=True)

    loss, _ = model.loss(prefix_embs)
    loss.backward()

    assert prefix_embs.grad is None
    encoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    )
    decoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.decoder.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad_norm > 0
    assert decoder_grad_norm > 0


class _FakeValueExpert:
    def __init__(self, image_emb, lang_emb):
        self.image_emb = image_emb
        self.lang_emb = lang_emb

    def embed_image(self, image):
        return self.image_emb.to(device=image.device)

    def embed_language_tokens(self, tokens):
        return self.lang_emb.to(device=tokens.device)


def _load_value_critic_model(monkeypatch):
    value_model_dir = (
        Path(__file__).resolve().parents[2]
        / "rlinf/models/embodiment/value_model/recap"
    )
    package_name = "value_model_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(value_model_dir)]
    monkeypatch.setitem(sys.modules, package_name, package)

    module_name = f"{package_name}.modeling_critic"
    spec = importlib.util.spec_from_file_location(
        module_name,
        value_model_dir / "modeling_critic.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module.ValueCriticModel


def test_value_model_does_not_rescale_gemma3_language_embeddings(monkeypatch):
    """Gemma3 embed_tokens already applies sqrt(hidden_size) internally."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.Gemma3ForCausalLM")

    ValueCriticModel = _load_value_critic_model(monkeypatch)

    hidden_size = 4
    image_emb = torch.zeros(1, 2, hidden_size)
    lang_emb = torch.arange(12, dtype=torch.float32).reshape(1, 3, hidden_size)

    model = SimpleNamespace(
        gradient_checkpointing_enabled=False,
        training=False,
        value_expert=_FakeValueExpert(image_emb=image_emb, lang_emb=lang_emb),
        _apply_checkpoint=lambda func, *args: func(*args),
    )

    prefix_embs, prefix_pad_masks = ValueCriticModel.embed_prefix(
        model,
        images=[torch.empty(1, 3, 8, 8)],
        img_masks=[torch.tensor([True])],
        lang_tokens=torch.tensor([[1, 2, 3]]),
        lang_masks=torch.tensor([[True, True, False]]),
    )

    torch.testing.assert_close(prefix_embs[:, 2:], lang_emb)
    torch.testing.assert_close(
        prefix_pad_masks,
        torch.tensor([[True, True, True, True, False]]),
    )


def _history_cfg():
    return OmegaConf.create(
        {
            "model": {
                "history_buffers": {
                    "main": {
                        "history_size": 2,
                        "min_history_size": 1,
                        "input_interval": 3,
                        "history_keys": ["main_images"],
                        "input_on_done": True,
                    }
                }
            }
        }
    )


def _append_step(manager: HistoryManager, value: int) -> None:
    manager.append_to_history_entries(
        {"main_images": torch.tensor([[value], [value + 10]])}
    )


def test_build_history_input_skips_between_interval_ticks():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_input == {}
    assert history_length == {}
    assert manager.history_counts == [2, 2]


def test_build_history_input_emits_on_interval_tick():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)
    _append_step(manager, 3)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_length == {"main": [2, 2]}
    assert history_input["main"]["main_images"][0] == [
        torch.tensor([2]),
        torch.tensor([3]),
    ]
    assert history_input["main"]["main_images"][1] == [
        torch.tensor([12]),
        torch.tensor([13]),
    ]


VALUE_CLIP = 0.2
HUBER_DELTA = 10.0


def _critic_metrics(values, prev_values, returns, loss_mask=None):
    _, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=loss_mask,
    )
    return metrics


def test_value_clip_ratio_is_zero_when_no_update_is_clipped():
    prev_values = torch.zeros(4, 8)
    values = torch.full((4, 8), VALUE_CLIP / 2)
    returns = torch.zeros(4, 8)

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_clip_ratio_reports_the_fraction_of_clipped_updates():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    # Half of the entries move outside the trust region, half stay inside.
    values = torch.full((4, 8), VALUE_CLIP / 2)
    values[:, :4] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_grows_with_the_size_of_the_value_update():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)

    ratios = [
        float(
            _critic_metrics(torch.full((4, 8), scale), prev_values, returns)[
                "critic/value_clip_ratio"
            ]
        )
        for scale in (0.5 * VALUE_CLIP, 2 * VALUE_CLIP)
    ]

    assert ratios == [pytest.approx(0.0), pytest.approx(1.0)]


def test_value_clip_ratio_ignores_masked_out_entries():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    loss_mask[:, :2] = True

    # Every valid entry is clipped; every padded entry is not.
    values = torch.zeros(4, 8)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(1.0)


def test_value_clip_ratio_broadcasts_a_narrower_loss_mask():
    prev_values = torch.zeros(4, 8, 3)
    returns = torch.zeros(4, 8, 3)
    loss_mask = torch.zeros(4, 8, 1, dtype=torch.bool)
    loss_mask[:, :4] = True

    values = torch.zeros(4, 8, 3)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    # 2 of the 4 unmasked steps are clipped.
    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_is_zero_when_every_entry_is_masked_out():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    values = torch.full((4, 8), 10 * VALUE_CLIP)

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_loss_is_unchanged_by_the_metric_computation():
    torch.manual_seed(0)
    prev_values = torch.randn(4, 8)
    values = torch.randn(4, 8, requires_grad=True)
    returns = torch.randn(4, 8)

    loss, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=None,
    )

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -VALUE_CLIP, VALUE_CLIP
    )
    expected = torch.max(
        torch.nn.functional.huber_loss(
            values, returns, delta=HUBER_DELTA, reduction="none"
        ),
        torch.nn.functional.huber_loss(
            value_pred_clipped, returns, delta=HUBER_DELTA, reduction="none"
        ),
    ).mean()

    assert float(loss.detach()) == pytest.approx(float(expected.detach()), abs=1e-6)
    assert loss.requires_grad
    assert not metrics["critic/value_clip_ratio"].requires_grad


def test_create_builds_expected_sampler_types():
    constant = DelaySampler.create(
        OmegaConf.create({"type": "constant", "delay": 0.12})
    )
    uniform = DelaySampler.create(
        OmegaConf.create({"type": "uniform", "min_delay": 0.03, "max_delay": 0.08})
    )
    exponential = DelaySampler.create(
        OmegaConf.create({"type": "exponential", "rate": 0.5})
    )
    gaussian = DelaySampler.create(
        OmegaConf.create({"type": "gaussian", "mean": 0.20, "stddev": 0.03})
    )

    assert isinstance(constant, ConstantDelaySampler)
    assert isinstance(uniform, UniformDelaySampler)
    assert isinstance(exponential, ExponentialDelaySampler)
    assert isinstance(gaussian, GaussianDelaySampler)


def test_create_accepts_none():
    assert DelaySampler.create(None) is None


def test_same_seed_produces_same_sequence_per_sampler():
    first = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)
    second = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)

    assert first.sample(8) == second.sample(8)


def test_constant_sampler_uses_seconds_helpers():
    sampler = ConstantDelaySampler(delay=0.25)

    assert sampler.sample(3) == [0.25, 0.25, 0.25]
    assert sampler.sample_one() == 0.25


def test_gaussian_sampler_never_returns_negative_seconds():
    sampler = GaussianDelaySampler(mean=0, stddev=0.1, seed=0)

    assert all(delay >= 0 for delay in sampler.sample(100))


def test_invalid_ranges_raise_clear_errors():
    with pytest.raises(ValueError, match="min_delay must be <="):
        UniformDelaySampler(min_delay=0.2, max_delay=0.1)

    with pytest.raises(ValueError, match="rate must be > 0"):
        ExponentialDelaySampler(rate=0)


def test_num_samples_must_be_non_negative_int():
    sampler = ConstantDelaySampler(delay=1)

    with pytest.raises(TypeError, match="num_samples must be an int"):
        sampler.sample(1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_samples must be >= 0"):
        sampler.sample(-1)


class _FakeEnv:
    """Minimal non-gym env exposing the chunk_step/reset surface."""

    def chunk_step(self, *args, **kwargs):
        return "stepped"

    def reset(self, *args, **kwargs):
        return "obs", {}


# Mock gymnasium and its transitive imports for unit-test environments that
# do not install the embodied extras. A minimal gym.Wrapper shim is enough
# because InsertDelay only delegates to self.env.


class _FakeGymEnv:
    pass


class _FakeGymWrapper:
    def __init__(self, env):
        self.env = env


_fake_gym = MagicMock()
_fake_gym.Env = _FakeGymEnv
_fake_gym.Wrapper = _FakeGymWrapper

if "gymnasium" not in sys.modules:
    sys.modules["gymnasium"] = _fake_gym
if "imageio" not in sys.modules:
    sys.modules["imageio"] = MagicMock()


def _delayed_env(delay: float):
    from rlinf.envs.wrappers import InsertDelay

    return InsertDelay(
        _FakeEnv(), OmegaConf.create({"type": "constant", "delay": delay})
    )


def test_chunk_step_does_not_block_the_caller():
    env = _delayed_env(0.5)

    start = time.monotonic()
    assert env.chunk_step() == "stepped"
    elapsed = time.monotonic() - start

    # The delay is sampled, not slept: blocking here would stall the event loop.
    assert elapsed < 0.05


def test_wait_delay_waits_out_the_accumulated_delay():
    env = _delayed_env(0.05)
    env.chunk_step()
    env.chunk_step()

    start = time.monotonic()
    asyncio.run(env.wait_delay())
    elapsed = time.monotonic() - start

    # Both sampled delays are paid, never dropped.
    assert elapsed == pytest.approx(0.1, abs=0.03)


def test_wait_delay_yields_to_other_coroutines():
    env = _delayed_env(0.2)
    env.chunk_step()
    progressed = []

    async def main():
        async def ticker():
            for _ in range(4):
                await asyncio.sleep(0.01)
                progressed.append(1)

        await asyncio.gather(env.wait_delay(), ticker())

    asyncio.run(main())
    # A blocking sleep would have starved the ticker entirely.
    assert len(progressed) == 4


def test_wait_delay_is_a_noop_when_nothing_is_pending():
    env = _delayed_env(0.5)

    start = time.monotonic()
    asyncio.run(env.wait_delay())

    assert time.monotonic() - start < 0.05


def test_delay_metrics_report_every_sample():
    env = _delayed_env(0.03)
    env.chunk_step()
    env.reset()

    metrics = env.insert_delay_metrics()

    assert metrics.tolist() == pytest.approx([0.03, 0.03])
    assert env.insert_delay_metrics().numel() == 0


def test_openwam_rl_forward_rescores_and_backpropagates():
    """The RL contract must produce finite scores and gradients on both heads."""

    class _Architecture(torch.nn.Module):
        action_dim = 4
        uses_proprioception = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.2))

        def forward(self, action, action_timestep, proprio=None, **inputs):
            del action_timestep, proprio
            return torch.zeros_like(
                inputs["latents"]
            ) + self.weight, action * 0 + self.weight

    engine = SimpleNamespace(
        architecture=_Architecture(),
        cfg=SimpleNamespace(
            dataloader=SimpleNamespace(multiview=False, camera_layout=[])
        ),
    )
    policy = OpenWAMPolicy(engine, num_frames=3, height=4, width=4, denoise_steps=2)
    forward_inputs = {
        "chains": torch.zeros(2, 2, 2, 4),
        "denoise_inds": torch.zeros(2, 2, dtype=torch.long),
        "video_latents": torch.zeros(2, 2, 3),
        "sigma": torch.ones(2, 1),
        "sigma_next": torch.zeros(2, 1),
        "noise_std": torch.full((2, 1), 0.1),
        "video_timesteps": torch.full((2, 1), 1000.0),
        "action_timesteps": torch.full((2, 1), 1000.0),
        "active_action_indices": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        "native__latents": torch.zeros(2, 2, 3),
        "native__context": torch.ones(2, 4, 3),
        "native__proprio": torch.ones(2, 8),
    }
    forward_inputs["chains"][:, 1] = 0.2
    result = policy.default_forward(forward_inputs=forward_inputs)
    assert result["logprobs"].shape == (2, 2, 4)
    assert result["values"].shape == (2,)
    (-result["logprobs"].mean() + result["values"].mean()).backward()
    assert policy.engine.architecture.weight.grad is not None
    assert policy.value_head[0].weight.grad is not None


# --- OpenWAM PPO checkpoint export -----------------------------------------


def _make_source(tmp_path: Path) -> Path:
    source = tmp_path / "openwam-sft"
    (source / "tokenizer").mkdir(parents=True)
    (source / "tokenizer" / "tokenizer.json").write_text("{}")
    (source / "config.yaml").write_text("model:\n  architecture: dual_system\n")
    np.save(source / "normalization_stats.npy", np.zeros(3, dtype=np.float32))
    save_file(
        {
            "action_backbone.weight": torch.zeros(2, 2),
            "video_backbone.bias": torch.zeros(2),
            "vlm_backbone.ignored": torch.zeros(1),
        },
        str(source / "checkpoint_step_30000.safetensors"),
    )
    return source


def _make_rlinf_checkpoint(tmp_path: Path, extra: dict | None = None) -> Path:
    step_dir = tmp_path / "results" / "checkpoints" / "global_step_7"
    (step_dir / "actor" / "model_state_dict").mkdir(parents=True)
    state = {
        "architecture.action_backbone.weight": torch.full((2, 2), 3.0),
        "architecture.video_backbone.bias": torch.full((2,), 4.0),
        "architecture.vlm_backbone.ignored": torch.ones(1),
        "value_head.0.weight": torch.ones(1, 8),
    }
    state.update(extra or {})
    torch.save(state, step_dir / "actor" / "model_state_dict" / "full_weights.pt")
    return step_dir


def test_openwam_export_rebuilds_native_checkpoint_dir(tmp_path):
    source = _make_source(tmp_path)
    step_dir = _make_rlinf_checkpoint(tmp_path)
    out = tmp_path / "openwam-ppo"

    written = export_checkpoint(step_dir, source, out)

    assert written == out / "checkpoint_step_7.safetensors"
    exported = load_file(str(written))
    assert set(exported) == {"action_backbone.weight", "video_backbone.bias"}
    assert torch.equal(exported["action_backbone.weight"], torch.full((2, 2), 3.0))
    assert (out / "config.yaml").read_text() == (source / "config.yaml").read_text()
    assert (out / "tokenizer" / "tokenizer.json").is_file()
    assert (out / "normalization_stats.npy").is_file()
    assert not list(out.glob("checkpoint_step_30000*"))
    value_head = torch.load(out / "rlinf_value_head.pt")
    assert set(value_head) == {"value_head.0.weight"}


def test_openwam_export_rejects_foreign_source(tmp_path):
    source = _make_source(tmp_path)
    step_dir = _make_rlinf_checkpoint(
        tmp_path, {"architecture.action_backbone.extra": torch.zeros(1)}
    )
    with pytest.raises(ValueError, match="differ from the source"):
        export_checkpoint(step_dir, source, tmp_path / "out")
    written = export_checkpoint(
        step_dir, source, tmp_path / "out", allow_key_mismatch=True
    )
    assert "action_backbone.extra" in load_file(str(written))


def test_openwam_export_links_assets_and_infers_step(tmp_path):
    source = _make_source(tmp_path)
    step_dir = _make_rlinf_checkpoint(tmp_path)
    assert infer_step(step_dir) == 7
    out = tmp_path / "linked"
    export_checkpoint(step_dir, source, out, step=12, link_assets=True)
    assert (out / "tokenizer").is_symlink()
    assert (out / "checkpoint_step_12.safetensors").is_file()
    # A bare weights file outside a global_step_N directory needs --step.
    bare = tmp_path / "weights.pt"
    bare.write_bytes(
        (step_dir / "actor" / "model_state_dict" / "full_weights.pt").read_bytes()
    )
    assert infer_step(bare) is None
    with pytest.raises(ValueError, match="--step"):
        export_checkpoint(bare, source, tmp_path / "nostep")
    written = export_checkpoint(bare, source, tmp_path / "bare", step=3)
    assert written.name == "checkpoint_step_3.safetensors"
