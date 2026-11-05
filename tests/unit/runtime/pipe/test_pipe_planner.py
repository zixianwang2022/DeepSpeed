# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import itertools
import json
import math
import random
import sys

import pytest
import torch
import torch.nn as nn

import deepspeed
import deepspeed.comm as dist
from deepspeed.runtime.pipe.planner import (StageModel, data_parallel_sizes, even_parts, plan, plan_memory_balanced,
                                            predict, profile_layer_types, write_costs, zero1_partition)
from deepspeed.runtime.pipe.planner.__main__ import main as planner_main
from unit.common import DistributedTest


def test_data_parallel_sizes_follow_deepspeed_groups():
    assert data_parallel_sizes(32, pp=4, tp=1, ep=8) == (8, 1)
    # without expert tensor parallelism (DeepSpeed's default), EP groups are consecutive ranks of a stage, whatever TP is
    assert data_parallel_sizes(32, pp=4, tp=2, ep=8) == (4, 1)
    assert data_parallel_sizes(32, pp=4, tp=8, ep=8) == (1, 1)
    with pytest.raises(ValueError):
        data_parallel_sizes(32, pp=4, tp=1, ep=16)  # 8 ranks per stage cannot hold EP 16
    # with expert tensor parallelism, EP groups are cut out of the data-parallel ranks
    assert data_parallel_sizes(32, pp=4, tp=2, ep=4, expert_tensor_parallel=True) == (4, 1)
    with pytest.raises(ValueError):
        data_parallel_sizes(32, pp=4, tp=8, ep=8, expert_tensor_parallel=True)


def _costs(sizes=(1, )):
    """A small model: Embed, six blocks of two kinds, Head. Recompute keeps an eighth of a block's memory and adds a
    forward. A micro-batch of size b takes b times the time and the memory of size 1."""

    def row(fwd, bwd, kept, numel):
        return {
            'param_list': [['w', numel, False]],
            'by_mbs': {
                str(b): {
                    'fwd_ms': b * fwd,
                    'bwd_ms': b * bwd,
                    'fwd_ckpt_ms': b * fwd,
                    'bwd_ckpt_ms': b * (bwd + fwd),
                    'kept_bytes': b * kept,
                    'kept_ckpt_bytes': b * kept // 8,
                    'transient_bytes': b * kept // 2,
                    'transient_ckpt_bytes': b * kept,
                    'stale_bytes': 0,
                    'stale_ckpt_bytes': 0,
                    'fwd_extra_bytes': 0,
                    'fwd_extra_ckpt_bytes': 0,
                    'kept_tensors': 0,
                    'kept_ckpt_tensors': 0,
                    'out_bytes': b << 20
                }
                for b in sizes
            }
        }

    return {
        'layers': {
            'Embed': row(1.0, 2.0, 1 << 22, 1 << 20),
            'A': row(3.0, 6.0, 1 << 26, 1 << 22),
            'B': row(2.0, 4.0, 1 << 25, 1 << 22),
            'Head': row(2.0, 3.0, 1 << 24, 1 << 20)
        }
    }


def _brute_force(model, budget, recompute=None):
    """Every split (cuts right before a block, at least one block per stage) whose stages fit. ``recompute=None``:
    each stage with the fewest recomputed blocks that fit, and the smallest time of the slowest stage. ``True`` or
    ``False``: every block or no block recomputed, and the smallest memory of the largest stage."""
    L = len(model.layers)
    cuts = [i for i in range(1, L) if model.layers[i] in model.blocks]
    best = math.inf
    for chosen in itertools.combinations(cuts, model.pp - 1):
        parts = [0, *chosen, L]
        stages = []
        for s in range(model.pp):
            n_blocks = sum(k in model.blocks for k in model.layers[parts[s]:parts[s + 1]])
            counts = range(n_blocks + 1) if recompute is None else [n_blocks if recompute else 0]
            fits = [
                st for c in counts for st in [model.stage(s, parts[s], parts[s + 1], c)] if st['reserve'] <= budget
            ]
            if n_blocks == 0 or not fits:
                break
            stages.append(fits[0])
        if len(stages) == model.pp:
            best = min(best, max(st['time_ms' if recompute is None else 'reserve'] for st in stages))
    return best


# The cases cover an even split without recompute, an even split with recompute on the first stage only, and two
# uneven splits with a different number of recomputed blocks on each stage.
@pytest.mark.parametrize("order, pp, budget_mib, layers_per_stage, recomputed",
                         [("AABABB", 2, 760, [3, 3], [0, 0]), ("AABABB", 3, 500, [2, 2, 2], [2, 0, 0]),
                          ("AAABBB", 2, 680, [2, 4], [0, 1]), ("AAABBB", 3, 510, [1, 2, 3], [0, 1, 1])])
def test_plan_matches_exhaustive_search(order, pp, budget_mib, layers_per_stage, recomputed):
    layers = ['Embed'] + list(order) + ['Head']
    model = StageModel(_costs(), layers, ['A', 'B'], pp=pp, dp=1, dp_e=1, m=8, b=1, bucket_elems=1e6)
    budget = budget_mib * 2**20
    best = plan(model, budget)
    assert best is not None
    assert (best['layers_per_stage'], best['activation_checkpoints_per_stage']) == (layers_per_stage, recomputed)
    assert best['max_stage_ms'] == pytest.approx(_brute_force(model, budget))
    stages, step = predict(model, best['parts'], best['activation_checkpoints_per_stage'])
    assert step == pytest.approx(best['step_ms'])
    assert all(st['reserve'] <= budget for st in stages)


def _random_model(rng):
    costs = _costs()
    for kind in ('A', 'B'):
        row = costs['layers'][kind]['by_mbs']['1']
        row['fwd_ms'] = row['fwd_ckpt_ms'] = rng.uniform(1, 9)
        row['bwd_ms'] = rng.uniform(1, 9)
        row['bwd_ckpt_ms'] = row['bwd_ms'] + row['fwd_ms']
        row['kept_bytes'] = rng.randrange(1 << 22, 1 << 27)
        row['kept_ckpt_bytes'] = row['kept_bytes'] // 8
    layers = ['Embed'] + [rng.choice('AB') for _ in range(rng.randrange(4, 9))] + ['Head']
    return StageModel(costs,
                      layers, ['A', 'B'],
                      pp=rng.choice([2, 3, 4]),
                      dp=1,
                      dp_e=1,
                      m=rng.choice([1, 2, 16]),
                      b=1,
                      bucket_elems=1e6)


def test_plan_makes_the_slowest_stage_fastest_on_random_costs():
    rng = random.Random(0)
    feasible = 0
    for _ in range(150):
        model = _random_model(rng)
        budget = rng.randrange(300, 900) * 2**20
        best, brute = plan(model, budget), _brute_force(model, budget)
        if best is None:
            assert brute == math.inf
        else:
            feasible += 1
            assert best['max_stage_ms'] == pytest.approx(brute)
            assert all(st['reserve'] <= budget for st in best['stages'])
    assert feasible > 50


@pytest.mark.parametrize("recompute", [True, False])
def test_memory_balanced_plan_makes_the_largest_stage_smallest_on_random_costs(recompute):
    rng = random.Random(1)
    feasible = 0
    for _ in range(150):
        model = _random_model(rng)
        budget = rng.randrange(300, 900) * 2**20
        best, brute = plan_memory_balanced(model, budget, recompute=recompute), _brute_force(model, budget, recompute)
        if best is None:
            assert brute == math.inf
        else:
            feasible += 1
            assert max(st['reserve'] for st in best['stages']) == brute
            wanted = best['layers_per_stage'] if recompute else [0] * model.pp
            assert best['activation_checkpoints_per_stage'] == wanted
    assert feasible > 30


class _Table:
    """A model whose stage times and memory come from a table, to state the search's rules without a costs file."""

    def __init__(self, blocks, pp, time_ms, m=4):
        self.layers, self.blocks, self.pp, self.m, self.time_ms = ['A'] * blocks, {'A'}, pp, m, time_ms

    def stage(self, s, lo, hi, c):
        return {'blocks': hi - lo, 'ckpt': c, 'time_ms': self.time_ms(s, hi - lo, c), 'reserve': hi - lo - c}

    step_ms, pass_ms, bubble = StageModel.step_ms, StageModel.pass_ms, StageModel.bubble
    overhead_ms, optimizer_ratio = 0.0, 0.0


def test_plan_keeps_the_earliest_cut_among_equally_fast_splits():
    """Stage 0 takes 10 ms with one, two or three blocks and the later stages less, so every split has the same
    slowest stage. For the last stage, and then for each stage before it, the dynamic program keeps the earliest cut
    among the equally good ones, so the last stages get the blocks."""
    best = plan(_Table(4, 2, lambda s, n, c: 10.0 if s == 0 else 5.0), budget_bytes=100)
    assert best['layers_per_stage'] == [1, 3] and best['max_stage_ms'] == 10.0
    best = plan(_Table(5, 3, lambda s, n, c: 10.0 if s == 0 else 5.0), budget_bytes=100)
    assert best['layers_per_stage'] == [1, 1, 3]


def test_plan_gives_each_stage_the_fewest_recomputed_blocks_that_fit():
    """A stage's memory is its blocks without recompute: with room for two, a stage of n blocks recomputes n - 2."""
    best = plan(_Table(8, 2, lambda s, n, c: n + c), budget_bytes=2)
    assert best['layers_per_stage'] == [4, 4] and best['activation_checkpoints_per_stage'] == [2, 2]
    assert plan(_Table(8, 2, lambda s, n, c: n + c), budget_bytes=-1) is None
    assert plan(_Table(1, 2, lambda s, n, c: n + c), budget_bytes=100) is None  # fewer blocks than stages
    # the fewest, not the fastest: here a recomputed block is faster, and a stage that fits still recomputes none
    best = plan(_Table(8, 2, lambda s, n, c: 10.0 * n - c), budget_bytes=100)
    assert best['layers_per_stage'] == [4, 4] and best['activation_checkpoints_per_stage'] == [0, 0]


def test_step_time_follows_the_1f1b_schedule():
    """Three stages of 10, 20 and 30 ms and 4 micro-batches. With 5 ms of overhead per stage and micro-batch the
    first micro-batch takes 15 + 25 + 35 ms and each of the other three follows 35 ms later: 180 ms. The optimizer
    step adds a tenth. The slowest stage works 4 x 35 of the 180 ms."""
    model = _Table(3, 3, lambda s, n, c: 10.0 * (s + 1))
    model.overhead_ms, model.optimizer_ratio = 5.0, 0.10
    best = plan(model, budget_bytes=100)
    assert model.pass_ms(best['stages']) == pytest.approx(180.0)
    assert best['step_ms'] == pytest.approx(198.0)
    assert best['bubble'] == pytest.approx(1 - 140.0 / 180.0)


def test_memory_balanced_step_time_takes_every_stage_as_slow_as_the_slowest():
    """ELM-PP reports a memory-balanced split with (m + pp - 1) slots of the slowest stage: 3 stages of 10, 20 and
    30 ms, 4 micro-batches and 5 ms of overhead give 6 x 35 = 210 ms, 231 ms with the optimizer step, and the
    slowest stage works 4 of the 6 slots."""
    model = _Table(3, 3, lambda s, n, c: 10.0 * (s + 1))
    model.overhead_ms, model.optimizer_ratio = 5.0, 0.10
    best = plan_memory_balanced(model, budget_bytes=100, recompute=False)
    assert best['step_ms'] == pytest.approx(231.0)
    assert best['bubble'] == pytest.approx(2 / 6)


def test_bubble_of_a_step_without_time_is_zero():
    model = _Table(2, 2, lambda s, n, c: 0.0)
    assert plan(model, budget_bytes=100)['bubble'] == 0.0


def test_even_split_gives_the_first_stages_one_more_block():
    model = StageModel(_costs(), ['Embed'] + ['A'] * 10 + ['Head'], ['A'], pp=4, dp=1, dp_e=1, m=1, b=1)
    parts = even_parts(model)
    assert parts == [0, 4, 7, 9, 12]
    assert [sum(k == 'A' for k in model.layers[lo:hi]) for lo, hi in zip(parts, parts[1:])] == [3, 3, 2, 2]


def test_reserve_lays_the_phases_side_by_side():
    """What a stage needs from the device is its 1F1B peak plus the large tensors of the reduce and of the step, which
    the 1F1B phase does not hold: a second bucket, the gradient partition, the flat copy of a gradient larger than
    the bucket (here every block's weight) and the copies of the largest group."""
    layers = ['Embed'] + list('AABABB') + ['Head']
    model = StageModel(_costs(), layers, ['A', 'B'], pp=2, dp=1, dp_e=1, m=8, b=1, bucket_elems=1e6)
    for s, (lo, hi) in enumerate([(0, 4), (4, 8)]):
        for c in range(4):
            st, states = model.stage(s, lo, hi, c), model.states(layers[lo:hi])
            # the bucket is doubled by default
            extra = model.B + states['G_part'] + states['oversize'] + states['step_group']
            assert st['reserve'] == st['mem']['1f1b'] + extra  # nothing is stale here, so the 1F1B peak leads
            assert st['reserve'] > st['peak'] == max(st['mem'].values())
    single = StageModel(_costs(),
                        layers, ['A', 'B'],
                        pp=2,
                        dp=1,
                        dp_e=1,
                        m=8,
                        b=1,
                        bucket_elems=1e6,
                        doubled_bucket=False)
    assert model.stage(0, 0, 4, 0)['reserve'] - single.stage(0, 0, 4, 0)['reserve'] == model.B


def test_reserve_adds_recompute():
    """The plan fits ``reserve``, not the allocated peak: a budget that the peaks fit but the reserve does not needs
    more recompute."""
    layers = ['Embed'] + list('AABABB') + ['Head']
    model = StageModel(_costs(), layers, ['A', 'B'], pp=2, dp=1, dp_e=1, m=8, b=1, bucket_elems=1e6)
    none = predict(model, [0, 4, 8], [0, 0])[0]
    budget = max(st['peak'] for st in none) + 2**20
    assert any(st['reserve'] > budget for st in none)
    best = plan(model, budget)
    assert best is not None and sum(best['activation_checkpoints_per_stage']) > 0
    assert all(st['reserve'] <= budget for st in best['stages'])


def test_last_stage_keeps_its_buffers_and_the_first_stage_its_inputs():
    """The last stage does not release a buffer after its backward: per buffer its output, the received activation
    and its gradient stay, and the engine keeps the loss of every micro-batch. The data of the micro-batches is on
    the first stage (inputs) and on the last (labels)."""
    layers = ['Embed'] + list('AABABB') + ['Head']
    costs, m = _costs(), 8
    kwargs = dict(pp=2, dp=1, dp_e=1, m=m, b=1, bucket_elems=1e6)
    base = StageModel(costs, layers, ['A', 'B'], **kwargs)
    first, last = base.stage(0, 0, 4, 0), base.stage(1, 4, 8, 0)
    out = costs['layers']['B']['by_mbs']['1']['out_bytes']  # what stage 1 receives, and the head's output
    states = base.states(layers[4:8])
    opt_and_params = states['params'] + states['opt']
    # reduce: parameters, optimizer state, gradients, both buckets, the gradient partition, the flat copy of the
    # largest gradient (larger than this small bucket), the receive buffer, and two buffers that each hold the
    # output, the received activation and its gradient
    assert last['mem']['reduce'] == (opt_and_params + states['grads'] + 2 * base.B + states['G_part'] +
                                     states['oversize'] + states['slack'] + out + 2 * (out + 2 * out))

    data = StageModel(costs, layers, ['A', 'B'], input_bytes=1 << 20, label_bytes=1 << 21, **kwargs)
    for phase in ('1f1b', 'reduce', 'step'):
        # two buffers with an input and its gradient each, and a copy for each of the two micro-batches in flight
        assert data.stage(0, 0, 4, 0)['mem'][phase] - first['mem'][phase] == (2 * 2 + 2) * (1 << 20)
        assert data.stage(1, 4, 8, 0)['mem'][phase] - last['mem'][phase] == 2 * (1 << 21)  # labels of two buffers

    costs['layers']['Head']['by_mbs']['1']['loss_bytes'] = 1 << 22
    costs['layers']['A']['buffer_bytes'] = 1 << 23
    more = StageModel(costs, layers, ['A', 'B'], **kwargs)
    for phase in ('1f1b', 'reduce', 'step'):
        assert more.stage(0, 0, 4, 0)['mem'][phase] - first['mem'][phase] == 2 * (1 << 23)  # two A layers
        assert more.stage(1, 4, 8, 0)['mem'][phase] - last['mem'][phase] == (1 << 23) + m * (1 << 22)


def test_reduce_holds_a_copy_of_a_gradient_larger_than_the_bucket():
    layers = ['Embed'] + list('AABABB') + ['Head']
    kwargs = dict(pp=2, dp=1, dp_e=1, m=8, b=1)
    big = (1 << 22) + 1  # the bucket holds the A and B weights (2^22 elements) but nothing larger
    fits, costs = StageModel(_costs(), layers, ['A', 'B'], bucket_elems=1 << 22, **kwargs), _costs()
    costs['layers']['Head']['param_list'] = [['w', big, False]]
    over = StageModel(costs, layers, ['A', 'B'], bucket_elems=1 << 22, **kwargs)
    assert fits.states(layers[4:8])['oversize'] == 0 and over.states(layers[4:8])['oversize'] == 2 * big
    a, b = fits.stage(1, 4, 8, 0), over.stage(1, 4, 8, 0)
    extra_params = big - (1 << 20)
    states_a, states_b = fits.states(layers[4:8]), over.states(layers[4:8])
    grown = sum(states_b[k] - states_a[k] for k in ('params', 'grads', 'opt', 'G_part'))
    assert grown >= 4 * extra_params
    assert b['mem']['reduce'] - a['mem']['reduce'] == grown + 2 * big


def _write_costs(tmp_path, ep=8):
    path = str(tmp_path / 'costs.json')
    write_costs(path, _costs()['layers'], {'device': 'test', 'tp': 1, 'ep': ep})
    return path


def _planner(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, 'argv', ['planner', *argv])
    try:
        planner_main()
        code = 0
    except SystemExit as e:
        code = e.code
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_command_line_plans_and_predicts(tmp_path, monkeypatch, capsys):
    costs = _write_costs(tmp_path)
    common = [
        '--costs', costs, '--layers', 'Embed,A*2,B,A,B*2,Head', '--blocks', 'A,B', '--world', '16', '--pp', '2',
        '--ep', '8', '--global-batch', '64', '--bucket-elems', '1e6'
    ]  # 8 data-parallel ranks, 8 micro-batches
    out_file = str(tmp_path / 'plan.json')
    code, out = _planner(monkeypatch, capsys, 'plan', *common, '--device-gib', '1.5', '--headroom-gib', '0.75',
                         '--out', out_file)
    assert code == 0
    best = json.loads([line for line in out.splitlines() if line.startswith('BEST ')][-1][5:])
    model = StageModel(_costs(), ['Embed'] + list('AABABB') + ['Head'], ['A', 'B'],
                       pp=2,
                       dp=8,
                       dp_e=1,
                       m=8,
                       b=1,
                       bucket_elems=1e6)
    budget = 0.75 * 2**30
    with open(out_file) as f:
        saved = json.load(f)[0]
    assert saved['layers_per_stage'] == best['layers_per_stage']
    assert saved['max_stage_ms'] == pytest.approx(_brute_force(model, budget))
    # the step: every stage once plus 7 times the slowest, each with 55 ms, and a tenth for the optimizer step
    times = [st['time_ms'] + 55.0 for st in saved['stages']]
    assert best['m'] == 8 and best['step_ms'] == pytest.approx((sum(times) + 7 * max(times)) * 1.10)
    # after the plan: the even split without and with full recompute, at the plan's micro-batch size. The first
    # budget holds both, the second (400 MiB) only the one with every block recomputed.
    unfit = 0
    for device_gib in ('1.5', '1.140625'):
        budget = (float(device_gib) - 0.75) * 2**30
        code, out = _planner(monkeypatch, capsys, 'plan', *common, '--device-gib', device_gib, '--headroom-gib',
                             '0.75')
        best = json.loads([line for line in out.splitlines() if line.startswith('BEST ')][-1][5:])
        for name, counts in (('no block recomputed', [0, 0]), ('every block recomputed', [3, 3])):
            line = next(x for x in out.splitlines() if x.startswith(f'EVEN SPLIT, {name}:'))
            stages, step = predict(model, even_parts(model), counts)
            over = [s for s, st in enumerate(stages) if st['reserve'] > budget]
            if over:
                unfit += 1
                assert line.endswith(f'stages {over} do not fit')
            else:
                assert f'step {step:.0f} ms, throughput {100 * (best["step_ms"] / step - 1):+.1f} %' in line
    assert unfit == 1
    budget = 0.75 * 2**30

    limits = ['--device-gib', '1.5', '--headroom-gib', '0.75']
    code, out = _planner(monkeypatch, capsys, 'predict', *common, *limits, '--parts', '0,4,8', '--ckpts', '1,0')
    assert code == 0 and 'recompute=[1, 0]' in out
    code, out = _planner(monkeypatch, capsys, 'predict', *common, *limits, '--ckpts', 'all')
    assert code == 0 and 'parts=[0, 4, 8] recompute=[3, 3]' in out and 'DOES NOT FIT' not in out
    code, out = _planner(monkeypatch, capsys, 'predict', *common, '--device-gib', '1.0', '--headroom-gib', '0.75')
    assert code == 0 and 'b=1: DOES NOT FIT, stages [0, 1] need more than the budget' in out

    code, out = _planner(monkeypatch, capsys, 'balance-memory', *common, *limits, '--recompute', 'all')
    assert code == 0
    best = json.loads([line for line in out.splitlines() if line.startswith('BEST ')][-1][5:])
    balanced = plan_memory_balanced(model, budget, recompute=True)
    assert best['layers_per_stage'] == balanced['layers_per_stage']
    assert best['activation_checkpoints_per_stage'] == best['layers_per_stage']

    # the two fixed times of the step: without them the step is the stage times alone
    code, out = _planner(monkeypatch, capsys, 'predict', *common, *limits, '--parts', '0,4,8', '--ckpts', '1,0',
                         '--overhead-ms', '0', '--optimizer-ratio', '0', '--out', out_file)
    with open(out_file) as f:
        saved = json.load(f)[0]
    times = [st['time_ms'] for st in saved['stages']]
    assert saved['step_ms'] == pytest.approx(sum(times) + 7 * max(times))


def _sized(tmp_path, monkeypatch, capsys, mode, *more, pp='2', world='16'):
    """The command line on a costs file with micro-batch sizes 1, 2 and 4; 8 data-parallel ranks, global batch 64."""
    path = str(tmp_path / 'costs_sizes.json')
    write_costs(path, _costs((1, 2, 4))['layers'], {'device': 'test', 'tp': 1, 'ep': 8})
    code, out = _planner(monkeypatch, capsys, mode, '--costs', path, '--layers', 'Embed,A*2,B,A,B*2,Head', '--blocks',
                         'A,B', '--world', world, '--pp', pp, '--ep', '8', '--global-batch', '64', '--bucket-elems',
                         '1e6', *more)
    best = [line for line in out.splitlines() if line.startswith('BEST ')]
    return code, out, json.loads(best[-1][5:]) if best and best[-1] != 'BEST none' else None


def _sized_model(b, pp=2, **kwargs):
    return StageModel(_costs((1, 2, 4)), ['Embed'] + list('AABABB') + ['Head'], ['A', 'B'],
                      pp=pp,
                      dp=8,
                      dp_e=1,
                      m=64 // (8 * b),
                      b=b,
                      bucket_elems=1e6,
                      **kwargs)


def test_command_line_takes_the_micro_batch_size_with_the_shortest_step(tmp_path, monkeypatch, capsys):
    """Without --mbs-list every size of the costs file is tried. The size with the shortest step wins."""
    limits = ['--device-gib', '3', '--headroom-gib', '0.75']
    budget = 2.25 * 2**30
    code, out, best = _sized(tmp_path, monkeypatch, capsys, 'plan', *limits)
    plans = {b: plan(_sized_model(b), budget) for b in (1, 2, 4)}
    assert all(plans.values())  # all three sizes fit, so the step time decides
    steps = {b: r['step_ms'] for b, r in plans.items()}
    assert len(set(steps.values())) == 3
    assert code == 0 and best['b'] == min(steps, key=steps.get) and best['step_ms'] == pytest.approx(
        min(steps.values()))
    # the order in which the sizes are given does not matter, and a size that does not divide the batch is skipped
    # before the costs file is asked for it
    code, out, again = _sized(tmp_path, monkeypatch, capsys, 'plan', *limits, '--mbs-list', '4,3,2,1')
    assert code == 0 and again == best and 'b=3: skipped' in out
    assert [x[:4] for x in out.splitlines() if x.startswith('b=') and ' m=' in x] == ['b=1 ', 'b=2 ', 'b=4 ']
    # a size that divides the batch and was not measured is an error
    code, out, _ = _sized(tmp_path, monkeypatch, capsys, 'plan', *limits, '--mbs-list', '1,8')
    assert code == 2 and 'does not hold micro-batch size 8' in out


def test_command_line_takes_the_smallest_size_among_equal_steps(tmp_path, monkeypatch, capsys):
    """One stage, no overhead: 8 micro-batches of size 1 take as long as 4 of size 2 or 2 of size 4."""
    more = ['--device-gib', '8', '--headroom-gib', '0.75', '--overhead-ms', '0', '--optimizer-ratio', '0']
    steps = {
        b: plan(_sized_model(b, pp=1, overhead_ms=0.0, optimizer_ratio=0.0), 7.25 * 2**30)['step_ms']
        for b in (1, 2, 4)
    }
    assert len(set(steps.values())) == 1
    code, out, best = _sized(tmp_path, monkeypatch, capsys, 'plan', *more, '--mbs-list', '4,2,1', pp='1', world='8')
    assert code == 0 and best['b'] == 1


def test_command_line_balances_memory_for_the_largest_size_that_fits(tmp_path, monkeypatch, capsys):
    need = {b: max(st['reserve'] for st in plan_memory_balanced(_sized_model(b), 2**40)['stages']) for b in (1, 2, 4)}
    assert need[1] < need[2] < need[4]
    for largest in (4, 2, 1):
        gib = need[largest] / 2**30 + 0.76  # just above what the largest size needs, with 0.75 GiB of headroom
        code, out, best = _sized(tmp_path, monkeypatch, capsys, 'balance-memory', '--device-gib', f'{gib:.6f}',
                                 '--headroom-gib', '0.75')
        assert code == 0 and best['b'] == largest
        assert best['activation_checkpoints_per_stage'] == best['layers_per_stage']
        # the step time of this mode: (m + pp - 1) slots of the slowest stage
        model = _sized_model(largest)
        want = plan_memory_balanced(model, (gib - 0.75) * 2**30)
        slowest = max(st['time_ms'] for st in want['stages']) + 55.0
        assert best['step_ms'] == pytest.approx((model.m + 1) * slowest * 1.10)


@pytest.mark.parametrize("change, message", [
    (['--layers', 'Embed,A*x,Head'], 'layer types separated by commas'),
    (['--mbs-list', '0'], 'positive micro-batch sizes'),
    (['--mbs-list', 'one'], 'micro-batch sizes separated by commas'),
    (['--max-group-size', 'many'], "a number of elements or 'none'"),
    (['--world', '12'], 'needs a multiple of 8 ranks per stage'),
    (['--parts', '0,3,6'], 'increasing layer boundaries from 0 to 8'),
    (['--parts', '2,4,8'], 'increasing layer boundaries from 0 to 8'),
    (['--parts', '0,1,8'], 'every stage needs a block'),
    (['--ckpts', '9,0'], 'between 0 and that number'),
    (['--blocks', 'Blok'], 'does not hold'),
    (['--mbs-list', '2'], 'does not hold micro-batch size 2'),
    (['--device-gib', '0.5', '--headroom-gib', '1'], 'larger than --headroom-gib'),
])
def test_command_line_rejects_what_it_cannot_model(tmp_path, monkeypatch, capsys, change, message):
    args = {
        '--costs': _write_costs(tmp_path),
        '--layers': 'Embed,A*2,B,A,B*2,Head',
        '--blocks': 'A,B',
        '--world': '16',
        '--pp': '2',
        '--ep': '8',
        '--global-batch': '64',
        '--device-gib': '1.5',
        '--headroom-gib': '0.75',
        '--parts': '0,4,8',
        '--ckpts': '1,0'
    }
    args.update(zip(change[::2], change[1::2]))
    code, out = _planner(monkeypatch, capsys, 'predict', *[v for kv in args.items() for v in kv])
    assert code == 2 and message in out


def test_command_line_refuses_costs_of_another_expert_parallel_size(tmp_path, monkeypatch):
    """A layer profiled at EP 8 holds an eighth of the experts: its costs say nothing about EP 4."""
    monkeypatch.setattr(sys, 'argv', [
        'planner', 'plan', '--costs',
        _write_costs(tmp_path, ep=8), '--layers', 'Embed,A*2,B,A,B*2,Head', '--blocks', 'A,B', '--world', '16', '--pp',
        '2', '--ep', '4', '--global-batch', '8', '--device-gib', '1.5'
    ])
    with pytest.raises(ValueError, match='another layout'):
        planner_main()


class TestZero1PartitionMatchesOptimizer(DistributedTest):
    world_size = 2

    def test_partition_sizes(self):
        """The calculator's ZeRO-1 partition sizes and gradient-partition copies equal those of DeepSpeed's optimizer."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(7, 5), nn.Linear(5, 11), nn.Linear(11, 3))
        groups = [list(model[0].parameters()) + list(model[1].parameters()), list(model[2].parameters())]
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 1
            },
            "zero_allow_untested_optimizer": True
        }
        engine, _, _, _ = deepspeed.initialize(config=config,
                                               model=model,
                                               optimizer=torch.optim.SGD([{
                                                   'params': g
                                               } for g in groups], lr=0.1))
        opt = engine.optimizer
        for i, g in enumerate(groups):
            ps, worst = zero1_partition([p.numel() for p in g], dp=2)
            assert opt.partition_size[i] == ps
            held = [None, None]
            dist.all_gather_object(held, sum(p.numel() for p in opt.params_in_partition[i]))
            assert max(held) == worst


class _Block(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(hidden, hidden, bias=False), nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return x + self.fc2(torch.tanh(self.fc1(x)))


class _Head(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.out(x)


class TestPredictedMemoryCoversRun(DistributedTest):
    """Profile the layer types, predict each stage, train with DeepSpeed's pipeline engine, ZeRO stage 1 and bf16, and
    compare phase by phase: no stage allocates more than predicted in the 1F1B schedule, in the gradient reduction or
    in the optimizer step. Uneven stages, recompute on the first stage only. The inputs and the labels are as large
    as an activation, and the loss is a scalar that lives in a buffer of that size, so the terms that a language
    model with token ids hardly sees count here."""
    world_size = 4

    def test_no_phase_allocates_more_than_predicted(self):
        from deepspeed.accelerator import get_accelerator
        from deepspeed.runtime.pipe import schedule
        from deepspeed.runtime.pipe.module import PipelineModule
        from deepspeed.runtime.pipe.topology import PipeDataParallelTopology
        acc = get_accelerator()
        if acc.device_name() == 'cpu':
            pytest.skip("needs a device with memory statistics")
        hidden, seq, blocks, micro_batches, bucket = 2048, 2048, 8, 4, 2**22
        layers_per_stage, recomputed = [3, 5], [2, 0]
        device, dtype = acc.current_device_name(), torch.bfloat16

        # the GEMM workspaces, as in the example's profile script: the first product of each thread leaves one
        acc.synchronize()
        before = acc.memory_allocated()
        lin = nn.Linear(hidden, hidden, bias=False).to(device, dtype)
        lin(torch.randn(8, hidden, device=device, dtype=dtype, requires_grad=True)).sum().backward()
        del lin
        acc.synchronize()
        workspace = acc.memory_allocated() - before

        loss_fn = nn.MSELoss()
        rows = profile_layer_types(
            {
                '_Block': lambda: _Block(hidden).to(device, dtype),
                '_Head': lambda: _Head(hidden).to(device, dtype)
            },
            lambda kind, b: torch.randn(b, seq, hidden, device=device, dtype=dtype, requires_grad=True), [1],
            reps=3,
            warmup=2,
            loss_fns={'_Head': lambda out: loss_fn(out.float(), torch.zeros_like(out, dtype=torch.float))},
            recompute={'_Block'},
            group=dist.get_world_group())
        costs = {'layers': rows, 'blas_workspace_bytes': workspace}

        config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": micro_batches,
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1,
                "reduce_bucket_size": bucket
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4,
                    "torch_adam": True
                }
            },
            "gradient_clipping": 1.0,
        }
        torch.manual_seed(0)
        module = PipelineModule(layers=[_Block(hidden) for _ in range(blocks)] + [_Head(hidden)],
                                topology=PipeDataParallelTopology(num_pp=2, num_dp=2),
                                loss_fn=lambda out, y: loss_fn(out.float(), y.float()),
                                partition_method='type:_Block',
                                activation_checkpoint_interval=1,
                                layers_per_stage=layers_per_stage,
                                activation_checkpoints_per_stage=recomputed)
        engine, _, _, _ = deepspeed.initialize(config=config, model=module, model_parameters=module.parameters())
        data = [(torch.randn(1, seq, hidden, dtype=dtype), torch.randn(1, seq, hidden, dtype=dtype))
                for _ in range(micro_batches)]

        # the peak of each phase: the engine runs the reduction and the step as schedule instructions
        peaks, phase = {'1f1b': 0, 'reduce': 0, 'step': 0}, ['1f1b']

        def close(next_phase):
            acc.synchronize()
            peaks[phase[0]] = max(peaks[phase[0]], acc.max_memory_allocated())
            acc.reset_peak_memory_stats()
            phase[0] = next_phase

        def starts(instruction, name):
            run = type(engine)._INSTRUCTION_MAP[instruction]

            def wrapped(self, *args, **kwargs):
                close(name)
                return run(self, *args, **kwargs)

            return wrapped

        engine._INSTRUCTION_MAP = {
            **type(engine)._INSTRUCTION_MAP, schedule.ReduceGrads: starts(schedule.ReduceGrads, 'reduce'),
            schedule.OptimizerStep: starts(schedule.OptimizerStep, 'step')
        }
        acc.synchronize()
        acc.reset_peak_memory_stats()
        for _ in range(3):
            engine.train_batch(data_iter=iter(data))
            close('1f1b')
        reserved = acc.max_memory_reserved()

        model = StageModel(costs, ['_Block'] * blocks + ['_Head'], ['_Block'],
                           pp=2,
                           dp=2,
                           dp_e=1,
                           m=micro_batches,
                           b=1,
                           bucket_elems=bucket,
                           max_group_size=None,
                           input_bytes=seq * hidden * 2,
                           label_bytes=seq * hidden * 2)
        stage_id = module.stage_id
        stage = model.stage(stage_id, module.parts[stage_id], module.parts[stage_id + 1], recomputed[stage_id])
        MiB = 2**20
        print(f"PLANNER_MEMORY stage {stage_id} rank {dist.get_rank()}, MiB, predicted / measured: " +
              ", ".join(f"{name} {stage['mem'][name] / MiB:.0f} / {peaks[name] / MiB:.0f}" for name in peaks) +
              f"; reserve {stage['reserve'] / MiB:.0f}, the allocator held {reserved / MiB:.0f}",
              flush=True)
        assert stage['blocks'] == layers_per_stage[stage_id]
        for name, measured in peaks.items():
            # Never too little, and not so much more that a plan would leave memory unused: every term takes the upper
            # end of what the code allows, and on this small model the widest gap seen (MI250X) was 20 %.
            assert 0 < measured <= stage['mem'][name] <= 1.25 * measured, name


class _WithTable(nn.Linear):

    def __init__(self, hidden):
        super().__init__(hidden, hidden)
        self.register_buffer('table', torch.zeros(3, hidden))


def test_profile_counts_saved_tensors_and_leaves_rng_state():
    hidden, b = 16, 4
    state = torch.get_rng_state()
    rows = profile_layer_types(
        {
            'Block': lambda: nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh()),
            'Head': lambda: _WithTable(hidden)
        },
        lambda kind, mbs: torch.randn(mbs, hidden, requires_grad=True),
        [b],
        reps=3,
        warmup=1,
        # a scalar that is a view into the buffer of the squares, as the mean of a loss can be
        loss_fns={'Head': lambda out: (out * out).view(-1)[0]},
        checkpoint_fn=lambda f, x: torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False))
    assert torch.equal(torch.get_rng_state(), state)
    r = rows['Block']['by_mbs'][str(b)]
    # Linear saves its input (the caller's) and its weight (a parameter); Tanh saves its output: only the output counts.
    assert r['kept_bytes'] == b * hidden * 4 and r['out_bytes'] == b * hidden * 4
    assert r['kept_tensors'] == 0 and r['fwd_extra_bytes'] == 0  # no tensor of 1 MiB; no memory statistics on the CPU
    assert r['fwd_ms']['median'] > 0 and r['bwd_ckpt_ms']['median'] > 0
    assert rows['Block']['param_list'] == [['0.weight', hidden * hidden, False], ['0.bias', hidden, False]]
    assert rows['Block']['buffer_bytes'] == 0 and r['loss_bytes'] == 0
    assert rows['Head']['buffer_bytes'] == 3 * hidden * 4
    assert rows['Head']['by_mbs'][str(b)]['loss_bytes'] == b * hidden * 4
