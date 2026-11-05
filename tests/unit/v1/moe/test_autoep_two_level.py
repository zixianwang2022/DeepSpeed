# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""The two-level transport must give the flat transport's output and gradients, at any micro-batch size, and put
fewer rows on the expert-parallel group."""

import copy
import socket

import deepspeed
import pytest
import torch
from deepspeed.module_inject.auto_ep_config import AutoEPConfig, validate_autoep_config
from deepspeed.moe.ep_two_level import _SumRowsByToken, _WeightedNodeSum
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import safe_get_full_grad
import deepspeed.comm as dist

from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import MockMoEOnlyTransformer, make_autoep_config, run_cpu_gloo_test, seed_everything

EXPERTS = 8
TOP_K = 4
HIDDEN = 64


def _model(num_experts=EXPERTS):
    model = MockMoEOnlyTransformer(num_layers=2, num_experts=num_experts, hidden_size=HIDDEN, intermediate_size=128)
    model.config.num_experts_per_tok = TOP_K
    return model


def _config(backend, ep_size, score_apply="auto", tp_size=1):
    config = make_autoep_config(zero_stage=0, ep_size=ep_size, mixed_precision=False)
    config["communication_data_type"] = "fp32"
    config["optimizer"]["params"]["torch_adam"] = True
    config["gradient_clipping"] = 0.0
    config["expert_parallel"].update(comm_backend=backend, score_apply=score_apply)
    if tp_size > 1:
        config["expert_parallel"]["sequence_sharded_moe"] = True
        config["tensor_parallel"] = {
            "autotp_size": tp_size,
            "partition_config": {
                "use_default_specs": False,
                "layer_specs": [{
                    "patterns": [".*\\.weight$"],
                    "partition_type": "skip",
                }],
            },
        }
    return config


class _RowsOverGroup:
    """Counts the token rows (of width HIDDEN) that this rank puts on all-to-all calls over groups of ``size`` ranks."""

    def __init__(self, size):
        self.size, self.rows, self.all_to_all_single = size, 0, dist.all_to_all_single

    def __enter__(self):
        dist.all_to_all_single = self
        return self

    def __exit__(self, *exc):
        dist.all_to_all_single = self.all_to_all_single

    def __call__(self, output, tensor, *args, **kwargs):
        group = kwargs.get("group", args[2] if len(args) > 2 else None)
        if tensor.dim() == 2 and tensor.shape[1] == HIDDEN and dist.get_world_size(group=group) == self.size:
            self.rows += tensor.shape[0]
        return self.all_to_all_single(output, tensor, *args, **kwargs)


def _run(state, config, batch, recompute=False):
    """Returns the output, the gradients, the rows this rank's forward put on the expert-parallel group, and for
    every MoE layer whether it took the two-level path."""
    model = _model()
    model.load_state_dict(state)
    engine = deepspeed.initialize(model=model, config=config)[0]
    inputs = batch.clone().to(engine.device, next(engine.module.parameters()).dtype)
    with _RowsOverGroup(config["expert_parallel"]["autoep_size"]) as counter:
        if recompute:
            # the backward runs the forward again, and with it every collective of the exchange
            output = torch.utils.checkpoint.checkpoint(engine.module, inputs, use_reentrant=False)
        else:
            output = engine(inputs)
    # a different weight on every element, so that a gradient that reached the wrong token would show
    probe = torch.sin(torch.arange(output.numel(), dtype=torch.float, device=output.device)).view(output.shape)
    engine.backward((output.float() * probe).mean())
    grads = {
        name: safe_get_full_grad(param).detach().float().cpu().clone()
        for name, param in engine.module.named_parameters()
        if param.requires_grad and safe_get_full_grad(param) is not None
    }
    two_level = [
        layer._two_level_topology is not None for layer in engine.module.modules()
        if getattr(layer, "_is_autoep_layer", False)
    ]
    return output.detach().float().cpu().clone(), grads, counter.rows, two_level


def _compare(rank,
             world_size,
             mbs,
             seqlen,
             group_size,
             score_apply,
             ep_size,
             tp_size=1,
             ranks_per_node=None,
             recompute=False):
    """``group_size`` > 0 gives every ``group_size`` consecutive ranks one host name, so that one host stands for
    several nodes; 0 keeps the host names the worker set. ``ranks_per_node``: the ranks of an expert-parallel group
    in one node (``group_size`` unless the host names decide)."""
    if group_size > 0:
        socket.gethostname = lambda: f"node{rank // group_size}"
    seed_everything(1234)
    state = copy.deepcopy(_model().state_dict())
    generator = torch.Generator().manual_seed(1000 + rank // tp_size)
    batch = torch.randn((mbs, seqlen, HIDDEN), generator=generator)
    flat_config = _config("comm", ep_size, score_apply, tp_size)
    flat_output, flat_grads, flat_rows, flat_layers = _run(state, flat_config, batch)
    two_output, two_grads, two_rows, two_layers = _run(state,
                                                       _config("two_level", ep_size, score_apply, tp_size),
                                                       batch,
                                                       recompute=recompute)
    # The comparison means nothing if the layers did not take the two-level path, or took it without sending less.
    ranks_per_node = ranks_per_node or group_size
    assert len(two_layers) == 2 and not any(flat_layers) and flat_rows > 0
    if ranks_per_node >= ep_size:
        assert not any(two_layers) and two_rows == flat_rows  # one node: the flat path
    else:
        assert all(two_layers)
        # 4 of 8 experts on 2 experts per rank: with two ranks or more in a node, a token's experts share a node
        assert two_rows < flat_rows if ranks_per_node > 1 else two_rows <= flat_rows
    errors = {"output": (two_output - flat_output).norm().item() / flat_output.norm().item()}
    assert flat_grads.keys() == two_grads.keys()
    for name in flat_grads:
        errors[name] = (two_grads[name] - flat_grads[name]).norm().item() / max(flat_grads[name].norm().item(), 1e-12)
    worst = max(errors, key=errors.get)
    if rank == 0:
        print(
            f"TWO_LEVEL_PARITY mbs={mbs} s={seqlen} ep={ep_size} tp={tp_size} group={group_size} "
            f"score={score_apply} params={len(flat_grads)} worst={worst} {errors[worst]:.2e} "
            f"rows on the group {two_rows} against {flat_rows}",
            flush=True)
    assert errors[worst] < 1e-4, f"{worst}: {errors[worst]}"


def _w_mbs1(rank, world, tmp):
    _compare(rank, world, 1, 32, 2, "auto", 4)


def _w_mbs2(rank, world, tmp):
    _compare(rank, world, 2, 32, 2, "auto", 4)


def _w_mbs4(rank, world, tmp):
    _compare(rank, world, 4, 32, 2, "auto", 4)


def _w_pre(rank, world, tmp):
    _compare(rank, world, 2, 32, 2, "pre", 4)


def _w_one_node(rank, world, tmp):
    _compare(rank, world, 2, 32, 4, "auto", 4)


def _w_every_rank_a_node(rank, world, tmp):
    _compare(rank, world, 2, 32, 1, "auto", 4)


def _w_uneven_nodes(rank, world, tmp):
    _compare(rank, world, 2, 32, 3, "auto", 4)  # a node of three ranks and a node of one


def _w_sharded(rank, world, tmp):
    _compare(rank, world, 2, 32, 2, "auto", 4, tp_size=2)


def _w_host_names(rank, world, tmp):
    # the host names as a real run sees them, here two ranks to a host
    socket.gethostname = lambda: f"node{rank // 2}"
    _compare(rank, world, 2, 32, 0, "auto", 4, ranks_per_node=2)


def _w_recompute(rank, world, tmp):
    _compare(rank, world, 2, 32, 2, "auto", 4, recompute=True)


@pytest.mark.parametrize("worker,world_size", [(_w_mbs1, 4), (_w_mbs2, 4), (_w_mbs4, 4), (_w_pre, 4), (_w_one_node, 4),
                                               (_w_every_rank_a_node, 4), (_w_uneven_nodes, 4), (_w_sharded, 8),
                                               (_w_host_names, 4), (_w_recompute, 4)])
def test_cpu_gloo_two_level_matches_flat(tmpdir, worker, world_size):
    run_cpu_gloo_test(worker, tmpdir, world_size=world_size)


class TestTwoLevelOnDevice(DistributedTest):
    """On the accelerator. In float32 the two-level exchange gives the flat exchange's output. In bf16, where an
    index_add with repeated indices adds in no fixed order, two runs of it give the same bits, and it agrees with
    the flat exchange to bf16 rounding."""
    world_size = 4

    def test_matches_flat_and_repeats_in_bf16(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("needs an accelerator")
        seed_everything(1234)
        state = copy.deepcopy(_model().state_dict())
        batch = torch.randn((4, 64, HIDDEN), generator=torch.Generator().manual_seed(1000 + dist.get_rank()))

        def config(backend, bf16=True):
            config = make_autoep_config(zero_stage=0, ep_size=4, mixed_precision=False)
            config["bf16"] = {"enabled": bf16}
            config["gradient_clipping"] = 0.0
            config["expert_parallel"].update(comm_backend=backend)
            return config

        # one host stands for two nodes of two GPUs each
        socket.gethostname = lambda: f"node{dist.get_rank() // 2}"

        exact_flat, exact_flat_grads, _, _ = _run(state, config("comm", bf16=False), batch)
        exact_two, exact_two_grads, _, exact_layers = _run(state, config("two_level", bf16=False), batch)
        assert all(exact_layers)
        assert (exact_two - exact_flat).norm().item() / exact_flat.norm().item() < 1e-4
        for name, grad in exact_flat_grads.items():
            assert (exact_two_grads[name] - grad).norm().item() / max(grad.norm().item(), 1e-12) < 1e-4, name

        flat_output, flat_grads, flat_rows, _ = _run(state, config("comm"), batch)
        first_output, first_grads, first_rows, first_layers = _run(state, config("two_level"), batch)
        second_output, second_grads, _, _ = _run(state, config("two_level"), batch)
        assert all(first_layers) and first_rows < flat_rows
        assert torch.equal(first_output, second_output)
        assert first_grads.keys() == second_grads.keys() == flat_grads.keys()
        assert all(torch.equal(first_grads[name], second_grads[name]) for name in first_grads)
        error = (first_output - flat_output).norm().item() / flat_output.norm().item()
        print(f"TWO_LEVEL_BF16 rank={dist.get_rank()} output error against flat {error:.2e}", flush=True)
        # bf16 keeps 8 bits, about 0.4 % per rounding, and the two paths round a token's sum at different places in
        # each of the two layers: a few percent at most. A row that reached the wrong token changes the output by
        # its own size.
        assert error < 5e-2


def _saved_by(build):
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        output = build()
    return output, saved


def test_sum_rows_by_token_matches_index_add_and_keeps_only_the_index():
    rows = torch.arange(6.0).view(3, 2).requires_grad_()
    index = torch.tensor([0, 0, 3])
    weight = torch.arange(1.0, 9.0).view(4, 2)
    output, saved = _saved_by(lambda: _SumRowsByToken.apply(rows, index, 4))
    assert len(saved) == 1 and saved[0] is index
    expected = torch.zeros(4, 2).index_add(0, index, rows)
    assert torch.equal(output, expected)
    assert torch.equal(*(torch.autograd.grad((out * weight).sum(), rows)[0] for out in (output, expected)))


def test_sum_rows_by_token_adds_in_float32_and_rounds_once():
    # 257 is not a bfloat16 number: adding four ones to 256 one at a time in bfloat16 stays at 256.
    rows = torch.tensor([[256.0], [1.0], [1.0], [1.0], [1.0]], dtype=torch.bfloat16)
    one_at_a_time = torch.zeros(1, dtype=torch.bfloat16)
    for row in rows:
        one_at_a_time = one_at_a_time + row
    assert one_at_a_time.item() == 256.0
    assert _SumRowsByToken.apply(rows, torch.zeros(5, dtype=torch.long), 1).item() == 260.0


@pytest.mark.parametrize("weighted", [True, False])
def test_weighted_node_sum_matches_autograd(weighted):
    torch.manual_seed(0)
    pilot, copy = torch.randn(4, 3, dtype=torch.double), torch.randn(5, 3, dtype=torch.double)
    pilot_weight, copy_weight = torch.rand(4, 1, dtype=torch.double), torch.rand(5, 1, dtype=torch.double)
    copy_row = torch.tensor([0, 0, 2, 3, 3])
    leaves = [t.requires_grad_() for t in (pilot, pilot_weight, copy, copy_weight)]

    def reference(pilot, pilot_weight, copy, copy_weight):
        if not weighted:
            return pilot.index_add(0, copy_row, copy)
        return (pilot * pilot_weight).index_add(0, copy_row, copy * copy_weight)

    def ours(pilot, pilot_weight, copy, copy_weight):
        if not weighted:
            return _WeightedNodeSum.apply(pilot, None, copy, None, copy_row)
        return _WeightedNodeSum.apply(pilot, pilot_weight, copy, copy_weight, copy_row)

    assert torch.allclose(ours(*leaves), reference(*leaves), rtol=0, atol=1e-12)
    probe = torch.randn(4, 3, dtype=torch.double)
    for mine, theirs in zip(torch.autograd.grad((ours(*leaves) * probe).sum(), leaves, allow_unused=True),
                            torch.autograd.grad((reference(*leaves) * probe).sum(), leaves, allow_unused=True)):
        assert (mine is None and theirs is None) or torch.allclose(mine, theirs, rtol=0, atol=1e-12)
    _, saved = _saved_by(lambda: ours(*leaves))
    assert len(saved) == (5 if weighted else 1)


@pytest.mark.parametrize("kwargs,tp_size,pp_size,message", [
    (dict(), 1, 2, "pipeline parallelism"),
    (dict(), 2, 1, "sequence_sharded_moe"),
    (dict(combine_impl="fused_weighted_sum"), 1, 1, "fused_weighted_sum"),
    (dict(combine_impl="legacy_bmm"), 1, 1, "legacy_bmm"),
    (dict(async_split_plan=True), 1, 1, "async_split_plan"),
])
def test_two_level_config_rejects_what_it_does_not_support(kwargs, tp_size, pp_size, message):
    config = AutoEPConfig(enabled=True, autoep_size=2, comm_backend="two_level", **kwargs)
    with pytest.raises(ValueError, match=message):
        validate_autoep_config(config, world_size=4 * pp_size, pp_size=pp_size, tp_size=tp_size, sp_size=1)
