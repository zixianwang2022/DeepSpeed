# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Gradient parity of the sequence-sharded AutoEP block against the unfolded baseline and assignment-level folding.
Every comparison also checks that the layers kept a shard of the tokens: with the option lost, the sharded run would
be a second folded run and every parity would hold."""

import copy
from types import SimpleNamespace

import deepspeed
import pytest
import torch

from deepspeed.module_inject.auto_ep import AutoEP
from deepspeed.module_inject.auto_ep_config import AutoEPConfig, validate_autoep_config
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.moe.ep_tp_dispatch import token_shard_counts
from deepspeed.utils import safe_get_full_grad
from unit.v1.moe.autoep_test_utils import (MockMoEOnlyTransformer, make_autoep_config, run_cpu_gloo_test,
                                           seed_everything, skip_unless_transformers_has, tiny_mixtral_config)
from unit.v1.moe.test_autoep_autotp_grad_parity import (GATE_BASELINE, GATE_FOLDED, INPUT_LAYERNORM,
                                                        _folded_zero0_tp2_ep4_config, _folded_zero0_tp4_ep4_config,
                                                        _folded_zero2_tp2_ep4_config, _full_grad_by_suffix,
                                                        _router_grad_model, _run_router_grad_boundary,
                                                        _zero0_baseline_config, _zero2_baseline_config)

CASES = {
    "zero0_tp2_ep4_world8": (_folded_zero0_tp2_ep4_config, _zero0_baseline_config, 2),
    "zero2_tp2_ep4_world8": (_folded_zero2_tp2_ep4_config, _zero2_baseline_config, 2),
    "zero0_tp2_ep4_world4": (_folded_zero0_tp2_ep4_config, _zero0_baseline_config, 2),
    "zero0_tp4_ep4_world4": (_folded_zero0_tp4_ep4_config, _zero0_baseline_config, 4),
}


class _TokenShards:
    """Records, for every forward of a sequence-sharded MoE layer on this peer, how many tokens entered the block and
    how many the peer kept."""

    def __enter__(self):
        self.seen, self.narrow = [], AutoEPMoELayer._narrow_token_shard
        recorder = self

        def narrow(layer, tokens):
            shard = recorder.narrow(layer, tokens)
            recorder.seen.append((tokens.shape[0], shard.shape[0]))
            return shard

        AutoEPMoELayer._narrow_token_shard = narrow
        return self

    def __exit__(self, *exc):
        AutoEPMoELayer._narrow_token_shard = self.narrow

    def assert_shards(self, tp_size):
        assert self.seen
        for entered, kept in self.seen:
            assert kept in token_shard_counts(entered, tp_size) and (kept < entered or entered == 1)


def _engine(reference_state, config):
    model = _router_grad_model()
    model.load_state_dict(reference_state)
    return deepspeed.initialize(model=model, config=config)[0]


def _run(engine, rank, world_size, tp_size, seed):
    _run_router_grad_boundary(engine,
                              logical_dp_world_size=world_size // tp_size,
                              logical_dp_rank=rank // tp_size,
                              seed=seed)


def _worker(rank, world_size, _shared_tmpdir, case):
    folded_config, baseline_config, tp_size = CASES[case]
    seed = 1234
    seed_everything(seed)
    reference_state = copy.deepcopy(_router_grad_model().state_dict())

    baseline = _engine(reference_state, baseline_config())
    _run(baseline, rank, world_size, tp_size, seed)
    baseline_gate = _full_grad_by_suffix(baseline, GATE_BASELINE)
    baseline_norm = _full_grad_by_suffix(baseline, INPUT_LAYERNORM)

    folded = _engine(reference_state, folded_config())
    with _TokenShards() as shards:
        _run(folded, rank, world_size, tp_size, seed)
    assert not shards.seen
    folded_experts = {name: _full_grad_by_suffix(folded, f".experts.{name}") for name in ("w1", "w2", "w3")}

    config = folded_config()
    config["expert_parallel"]["sequence_sharded_moe"] = True
    sharded = _engine(reference_state, config)
    with _TokenShards() as shards:
        _run(sharded, rank, world_size, tp_size, seed)
    shards.assert_shards(tp_size)
    gate = _full_grad_by_suffix(sharded, GATE_FOLDED)
    norm = _full_grad_by_suffix(sharded, INPUT_LAYERNORM)
    experts = {name: _full_grad_by_suffix(sharded, f".experts.{name}") for name in ("w1", "w2", "w3")}

    def error(actual, expected):
        return (actual - expected).norm().item() / max(expected.norm().item(), 1e-12)

    errors = {"gate": error(gate, baseline_gate), "norm": error(norm, baseline_norm)}
    errors.update({f"expert_{name}": error(experts[name], folded_experts[name]) for name in experts})
    if rank == 0:
        print(f"SEQUENCE_SHARDED_PARITY {case} {errors}", flush=True)
    for name, value in errors.items():
        assert value < 5e-3, f"{case} {name} relative error {value}"


def _worker_zero0_tp2_world8(rank, world_size, tmp):
    _worker(rank, world_size, tmp, "zero0_tp2_ep4_world8")


def _worker_zero2_tp2_world8(rank, world_size, tmp):
    _worker(rank, world_size, tmp, "zero2_tp2_ep4_world8")


def _worker_zero0_tp2_world4(rank, world_size, tmp):
    _worker(rank, world_size, tmp, "zero0_tp2_ep4_world4")


def _worker_zero0_tp4_world4(rank, world_size, tmp):
    _worker(rank, world_size, tmp, "zero0_tp4_ep4_world4")


@pytest.mark.parametrize("worker,world_size", [(_worker_zero0_tp2_world8, 8), (_worker_zero2_tp2_world8, 8),
                                               (_worker_zero0_tp2_world4, 4), (_worker_zero0_tp4_world4, 4)])
def test_cpu_gloo_sequence_sharded_grad_parity(tmpdir, worker, world_size):
    run_cpu_gloo_test(worker, tmpdir, world_size=world_size)


def test_sequence_sharded_config_rejects_what_it_does_not_support():
    config = AutoEPConfig(enabled=True, autoep_size=2, sequence_sharded_moe=True)
    with pytest.raises(ValueError, match="autotp_size"):
        validate_autoep_config(config, world_size=4, pp_size=1, tp_size=1, sp_size=1)
    config = AutoEPConfig(enabled=True, autoep_size=2, sequence_sharded_moe=True, comm_backend="deepep")
    with pytest.raises(ValueError, match="deepep"):
        validate_autoep_config(config, world_size=4, pp_size=1, tp_size=2, sp_size=1)


def test_sequence_sharded_rejects_router_logits_without_a_gather_path():
    """A layer whose preset neither returns its router logits nor points Transformers' recorder at the gathered ones
    cannot give the auxiliary loss every token's logits: output_router_logits is refused for it, at conversion and at
    every call. A layer with such a path is not refused."""
    config = AutoEPConfig(enabled=True, autoep_size=2, sequence_sharded_moe=True)

    class Model(torch.nn.Linear):

        def forward(self, x, output_router_logits=None):
            return super().forward(x)

    def spec(target, returns=False, adapter="default"):
        return SimpleNamespace(moe_module_name=f"layer_{target}_{returns}_{adapter}",
                               router_logits_capture_target=target,
                               return_router_logits=returns,
                               preset_adapter=adapter)

    without = spec("none")
    with_paths = [spec("router", returns=True), spec("router", adapter="mixtral")]

    # set on the model's config, or on the text config of a model that has one
    for model_config in (SimpleNamespace(output_router_logits=True),
                         SimpleNamespace(get_text_config=lambda: SimpleNamespace(output_router_logits=True))):
        model = Model(1, 1)
        model.config = model_config
        AutoEP(model, config)._check_sequence_sharded_router_logits(with_paths)
        with pytest.raises(ValueError, match="layer_none"):
            AutoEP(model, config)._check_sequence_sharded_router_logits(with_paths + [without])

    # asked for at a call, or set on the config after the model was converted
    model = Model(1, 1)
    model.config = SimpleNamespace(output_router_logits=False)
    AutoEP(model, config)._check_sequence_sharded_router_logits([without])
    model(torch.zeros(1, 1))
    with pytest.raises(RuntimeError, match="output_router_logits"):
        model(torch.zeros(1, 1), output_router_logits=True)
    model.config.output_router_logits = True
    with pytest.raises(RuntimeError, match="output_router_logits"):
        model(torch.zeros(1, 1))


def _mixtral_with_aux_loss(rank, world_size, _shared_tmpdir, checkpointing=False):
    """A Hugging Face Mixtral with output_router_logits: the router auxiliary loss enters the loss. Folding routes every
    token on every peer, so its recorder sees every token's logits; sequence sharding must give the same auxiliary
    loss, loss and router gradient, and must have kept a shard of the tokens. With ``checkpointing`` every decoder
    layer is recomputed in the backward (Transformers' default, non-reentrant), where no recorder is active: the
    recompute must repeat the forward's work."""
    import transformers
    seed_everything(1234)
    model_config = tiny_mixtral_config(transformers)
    model_config.router_aux_loss_coef = 0.5  # large, so that a wrong auxiliary loss shows in the loss and gradient
    state = copy.deepcopy(transformers.MixtralForCausalLM(model_config).state_dict())
    ids = torch.randint(0, model_config.vocab_size, (1, 16), generator=torch.Generator().manual_seed(1000 + rank // 2))

    def run(sequence_sharded):
        config = make_autoep_config(zero_stage=0, ep_size=4, mixed_precision=False)
        config["communication_data_type"] = "fp32"
        config["optimizer"]["params"]["torch_adam"] = True
        config["gradient_clipping"] = 0.0
        config["expert_parallel"]["sequence_sharded_moe"] = sequence_sharded
        config["tensor_parallel"] = {
            "autotp_size": 2,
            "partition_config": {
                "use_default_specs": False,
                "layer_specs": [{
                    "patterns": [".*\\.weight$"],
                    "partition_type": "skip",
                }],
            },
        }
        model = transformers.MixtralForCausalLM(model_config)
        model.load_state_dict(state)
        if checkpointing:
            model.gradient_checkpointing_enable()
        engine = deepspeed.initialize(model=model, config=config)[0]
        with _TokenShards() as shards:
            out = engine(input_ids=ids.to(engine.device), labels=ids.to(engine.device), output_router_logits=True)
        engine.backward(out.loss)
        gate = next(
            safe_get_full_grad(p) for name, p in engine.module.named_parameters()
            if name.endswith("mlp.router.gate.weight"))
        return out.loss.item(), out.aux_loss.item(), gate.detach().float().cpu().clone(), shards

    folded_loss, folded_aux, folded_gate, folded_shards = run(False)
    loss, aux, gate, shards = run(True)
    assert not folded_shards.seen
    shards.assert_shards(2)
    gate_error = (gate - folded_gate).norm().item() / folded_gate.norm().item()
    if rank == 0:
        print(
            f"SEQUENCE_SHARDED_AUX_LOSS checkpointing={checkpointing} aux {aux:.6f} / {folded_aux:.6f} loss {loss:.6f} / {folded_loss:.6f} "
            f"gate {gate_error:.2e}",
            flush=True)
    assert abs(aux - folded_aux) <= 1e-5 * abs(folded_aux)
    assert abs(loss - folded_loss) <= 1e-5 * abs(folded_loss)
    assert gate_error < 1e-4


def _mixtral_with_aux_loss_recomputed(rank, world_size, tmp):
    _mixtral_with_aux_loss(rank, world_size, tmp, checkpointing=True)


@pytest.mark.parametrize("worker", [_mixtral_with_aux_loss, _mixtral_with_aux_loss_recomputed])
def test_cpu_gloo_sequence_sharded_router_aux_loss(tmpdir, worker):
    transformers = pytest.importorskip("transformers")
    skip_unless_transformers_has(transformers,
                                 "MixtralConfig",
                                 "MixtralForCausalLM",
                                 min_version="5.0.0",
                                 reason="Mixtral router-logit recording")
    run_cpu_gloo_test(worker, tmpdir, world_size=4)


def test_sequence_sharded_config_rejects_the_folding_check():
    config = AutoEPConfig(enabled=True, autoep_size=2, sequence_sharded_moe=True, validate_folding_routing=True)
    with pytest.raises(ValueError, match="validate_folding_routing"):
        validate_autoep_config(config, world_size=4, pp_size=1, tp_size=2, sp_size=1)


def _output_and_grads(state, config, batch):
    model = MockMoEOnlyTransformer(num_layers=2, num_experts=8, hidden_size=batch.shape[-1], intermediate_size=128)
    model.config.num_experts_per_tok = 4
    model.load_state_dict(state)
    engine = deepspeed.initialize(model=model, config=config)[0]
    with _TokenShards() as shards:
        output = engine(batch.clone().to(engine.device))
    # a different weight on every element, so that a gradient that reached the wrong token would show
    probe = torch.sin(torch.arange(output.numel(), dtype=torch.float, device=output.device)).view(output.shape)
    engine.backward((output.float() * probe).mean())
    grads = {
        name: safe_get_full_grad(param).detach().float().cpu().clone()
        for name, param in engine.module.named_parameters()
        if param.requires_grad and safe_get_full_grad(param) is not None
    }
    return output.detach().float().cpu().clone(), grads, shards


def _sharded_matches_folded(rank, mbs, seqlen):
    """The same weights and batch through folding and through sequence sharding, TP 2 x EP 4."""
    hidden = 64
    seed_everything(1234)
    reference = MockMoEOnlyTransformer(num_layers=2, num_experts=8, hidden_size=hidden, intermediate_size=128)
    state = copy.deepcopy(reference.state_dict())
    batch = torch.randn((mbs, seqlen, hidden), generator=torch.Generator().manual_seed(1000 + rank // 2))
    folded = make_autoep_config(zero_stage=0, ep_size=4, mixed_precision=False)
    folded["communication_data_type"] = "fp32"
    folded["optimizer"]["params"]["torch_adam"] = True
    folded["gradient_clipping"] = 0.0
    folded["tensor_parallel"] = {
        "autotp_size": 2,
        "partition_config": {
            "use_default_specs": False,
            "layer_specs": [{
                "patterns": [".*\\.weight$"],
                "partition_type": "skip",
            }],
        },
    }
    sharded = copy.deepcopy(folded)
    sharded["expert_parallel"]["sequence_sharded_moe"] = True
    folded_output, folded_grads, folded_shards = _output_and_grads(state, folded, batch)
    sharded_output, sharded_grads, shards = _output_and_grads(state, sharded, batch)
    assert not folded_shards.seen and len(shards.seen) == 2  # both layers kept a shard
    shards.assert_shards(2)
    errors = {"output": (sharded_output - folded_output).norm().item() / folded_output.norm().item()}
    assert folded_grads.keys() == sharded_grads.keys()
    for name in folded_grads:
        errors[name] = (sharded_grads[name] - folded_grads[name]).norm().item() / max(
            folded_grads[name].norm().item(), 1e-12)
    worst = max(errors, key=errors.get)
    if rank == 0:
        print(f"SEQUENCE_SHARDED_TOKENS tokens={mbs * seqlen} worst={worst} {errors[worst]:.2e}", flush=True)
    assert errors[worst] < 5e-3, f"{worst}: {errors[worst]}"


def _worker_odd_token_count(rank, world_size, tmp):
    _sharded_matches_folded(rank, 1, 33)  # shards of 17 and 16 tokens


def _worker_fewer_tokens_than_peers(rank, world_size, tmp):
    _sharded_matches_folded(rank, 1, 1)  # one peer holds the token, the other an empty shard


@pytest.mark.parametrize("worker", [_worker_odd_token_count, _worker_fewer_tokens_than_peers])
def test_cpu_gloo_sequence_sharded_uneven_token_shards(tmpdir, worker):
    run_cpu_gloo_test(worker, tmpdir, world_size=8)
