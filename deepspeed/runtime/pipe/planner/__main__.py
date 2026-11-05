# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""python -m deepspeed.runtime.pipe.planner <mode> --costs <file> --layers 'Embed,Block*32,Head' --blocks Block ...

  plan            the split whose slowest stage is fastest, for every micro-batch size, and the size with the
                  smallest step time; then the even split without and with full recompute at that size
  predict         the memory and the step time of a given split (--parts, --ckpts); the even split by default
  balance-memory  the split whose largest stage memory is smallest, with every block recomputed (--recompute all)
                  or none, and the largest micro-batch size for which one fits

The micro-batch sizes are those of --mbs-list, or every size that the costs file holds. A size whose data-parallel
batch does not divide --global-batch is skipped.
"""

import argparse
import json

from .memory import GiB, StageModel, data_parallel_sizes
from .profile import load_costs
from .search import even_parts, plan, plan_memory_balanced, predict


def layer_list(spec):
    """'Embed,Block*32,Head' -> ['Embed', 'Block', ..., 'Head']"""
    out = []
    for t in spec.split(','):
        name, _, n = t.partition('*')
        if not name or (n and int(n) < 1):
            raise ValueError(t)
        out += [name] * (int(n) if n else 1)
    return out


def main():
    ap = argparse.ArgumentParser(prog='python -m deepspeed.runtime.pipe.planner')
    ap.add_argument('mode', choices=['plan', 'predict', 'balance-memory'])
    ap.add_argument('--costs', required=True)
    ap.add_argument('--layers', required=True, help="the layer types in order, e.g. 'Embed,Block*32,Head'")
    ap.add_argument('--blocks', default='Block', help='the layer types that are counted and may be recomputed')
    ap.add_argument('--world', type=int, required=True, help='GPUs of the run')
    ap.add_argument('--pp', type=int, required=True)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--ep', type=int, default=1)
    ap.add_argument('--expert-tensor-parallel',
                    action='store_true',
                    help="the MoE layers' enable_expert_tensor_parallelism")
    ap.add_argument('--global-batch', type=int, required=True, help='samples per step')
    ap.add_argument('--mbs-list',
                    default='',
                    help='micro-batch sizes to compare, e.g. 1,2,4 (default: every size that the costs file holds)')
    ap.add_argument('--device-gib', type=float, required=True)
    ap.add_argument('--headroom-gib',
                    type=float,
                    default=4.0,
                    help="device memory the plan may not use: what the device holds outside PyTorch's allocator (its "
                    'context, the communication library; the used memory the device reports minus '
                    'torch.cuda.memory_reserved()) plus a margin')
    ap.add_argument('--bucket-elems', type=float, default=5e8, help='zero_optimization.reduce_bucket_size')
    ap.add_argument('--max-group-size', default='178956971', help="'none' = one expert group")
    ap.add_argument('--single-bucket', action='store_true', help='DeepSpeed allocates the pipeline reduce bucket once')
    ap.add_argument('--inplace-optimizer',
                    action='store_true',
                    help='the optimizer updates in place (DeepSpeed FusedAdam): the step holds 8 bytes per element of '
                    "the largest group partition instead of the 12 of torch.optim's Adam and AdamW")
    ap.add_argument('--input-bytes',
                    type=int,
                    default=0,
                    help="bytes of one micro-batch's inputs on the device (token ids are negligible)")
    ap.add_argument('--label-bytes', type=int, default=0, help="bytes of one micro-batch's labels on the device")
    ap.add_argument('--overhead-ms',
                    type=float,
                    default=55.0,
                    help='time that every stage adds to each micro-batch outside its layers (the default is the '
                    "value X-MoE-4D's planner uses)")
    ap.add_argument('--optimizer-ratio',
                    type=float,
                    default=0.10,
                    help="the optimizer step as a share of the forward and backward passes of a step (the default is "
                    "the value X-MoE-4D's planner uses)")
    ap.add_argument('--parts',
                    default='',
                    help='predict: layer boundaries, e.g. 0,9,17,25,34 (default: the even split)')
    ap.add_argument('--ckpts',
                    default='none',
                    help="predict: recomputed blocks per stage, e.g. 5,4,1,0, or 'all' or 'none'")
    ap.add_argument('--recompute',
                    choices=['all', 'none'],
                    default='all',
                    help='balance-memory: the blocks recomputed')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    try:
        layers = layer_list(a.layers)
    except ValueError:
        ap.error("--layers takes layer types separated by commas, each with an optional count: 'Embed,Block*32,Head'")
    costs, blocks = load_costs(a.costs, expect={'tp': a.tp, 'ep': a.ep}), a.blocks.split(',')
    missing = sorted(set(layers + blocks) - set(costs['layers']))
    if missing:
        ap.error(f'--layers or --blocks name {missing}, which {a.costs} does not hold (it holds '
                 f'{sorted(costs["layers"])})')
    if not set(blocks) & set(layers):
        ap.error(f'--layers holds no layer of the types in --blocks ({blocks})')
    rows = {kind: costs['layers'][kind]['by_mbs'] for kind in sorted(set(layers))}
    measured = sorted(set.intersection(*[{int(b) for b in r} for r in rows.values()]))
    try:
        sizes = sorted({int(v) for v in a.mbs_list.split(',')}) if a.mbs_list else measured
    except ValueError:
        ap.error('--mbs-list takes micro-batch sizes separated by commas')
    if not sizes or sizes[0] < 1:
        ap.error(f'--mbs-list takes positive micro-batch sizes, got {sizes}')
    try:
        mgs = None if a.max_group_size == 'none' else int(float(a.max_group_size))
    except ValueError:
        ap.error("--max-group-size takes a number of elements or 'none'")
    if a.device_gib <= a.headroom_gib:
        ap.error('--device-gib must be larger than --headroom-gib')
    if sum(k in blocks for k in layers) < a.pp:
        ap.error(f'--layers holds fewer blocks than --pp {a.pp} stages; every stage needs a block')
    parts = ckpts = None
    if a.mode == 'predict':
        try:
            parts = [int(v) for v in a.parts.split(',')] if a.parts else None
            ckpts = a.ckpts if a.ckpts in ('all', 'none') else [int(v) for v in a.ckpts.split(',')]
        except ValueError:
            ap.error("--parts takes integers separated by commas, --ckpts also 'all' or 'none'")
        if parts is not None:
            if len(parts) != a.pp + 1 or parts[0] != 0 or parts[-1] != len(layers) or sorted(set(parts)) != parts:
                ap.error(f'--parts needs {a.pp + 1} increasing layer boundaries from 0 to {len(layers)}, got {parts}')
            in_stage = [sum(k in blocks for k in layers[lo:hi]) for lo, hi in zip(parts, parts[1:])]
            if 0 in in_stage:
                ap.error(f'the stages of --parts hold {in_stage} blocks; every stage needs a block')
    try:
        dp, dp_e = data_parallel_sizes(a.world, a.pp, a.tp, a.ep, a.expert_tensor_parallel)
    except ValueError as e:
        ap.error(str(e))
    budget = (a.device_gib - a.headroom_gib) * GiB
    results, models = [], {}

    def given(model, parts, ckpts):
        in_stage = [sum(k in blocks for k in layers[lo:hi]) for lo, hi in zip(parts, parts[1:])]
        counts = in_stage if ckpts == 'all' else [0] * a.pp if ckpts == 'none' else ckpts
        if len(counts) != a.pp or any(c < 0 or c > n for c, n in zip(counts, in_stage)):
            ap.error(f'the stages hold {in_stage} blocks; --ckpts needs one count per stage between 0 and that '
                     f'number, got {counts}')
        stages, step = predict(model, parts, counts)
        return {
            'parts': parts,
            'layers_per_stage': in_stage,
            'activation_checkpoints_per_stage': counts,
            'stages': stages,
            'step_ms': step,
            'max_stage_ms': max(st['time_ms'] for st in stages),
            'bubble': model.bubble(stages)
        }

    def show(head, r):
        print(f"{head} parts={r['parts']} recompute={r['activation_checkpoints_per_stage']} "
              f"step {r['step_ms']:.0f} ms, slowest stage {r['max_stage_ms']:.1f} ms, bubble "
              f"{100 * r['bubble']:.1f} %, optimizer groups per stage {[st['groups'] for st in r['stages']]}")
        for s, st in enumerate(r['stages']):
            mm = st['mem']
            print(
                f"  stage {s}: {st['blocks']} blocks, {st['ckpt']} recomputed, {st['time_ms']:.1f} ms per micro-batch; "
                f"GiB: init {mm['after_init'] / GiB:.2f}, 1F1B {mm['1f1b'] / GiB:.2f}, reduce "
                f"{mm['reduce'] / GiB:.2f}, step {mm['step'] / GiB:.2f}; needs from the device "
                f"{st['reserve'] / GiB:.2f}" + (' (more than the budget)' if st['reserve'] > budget else ''))

    for b in sizes:
        if a.global_batch % (b * dp):
            print(f'b={b}: skipped, --global-batch {a.global_batch} is not a multiple of {b} x {dp} data-parallel '
                  f'ranks')
            continue
        lacking = [kind for kind, r in rows.items() if str(b) not in r]
        if lacking:
            ap.error(f'{a.costs} does not hold micro-batch size {b} for {lacking}; every layer type has the '
                     f'micro-batch sizes {measured}')
        if any('kept_ckpt_bytes' not in r[str(b)] for kind, r in rows.items() if kind in blocks):
            ap.error(f'{a.costs} holds a type of --blocks without recompute; profile it with recompute')
        model = StageModel(costs,
                           layers,
                           blocks,
                           a.pp,
                           dp,
                           dp_e,
                           a.global_batch // (b * dp),
                           b,
                           bucket_elems=a.bucket_elems,
                           max_group_size=mgs,
                           doubled_bucket=not a.single_bucket,
                           step_bytes=8 if a.inplace_optimizer else 12,
                           input_bytes=a.input_bytes,
                           label_bytes=a.label_bytes,
                           overhead_ms=a.overhead_ms,
                           optimizer_ratio=a.optimizer_ratio)
        models[b] = model
        if a.mode == 'predict':
            r = given(model, parts or even_parts(model), ckpts)
            r['fits'] = all(st['reserve'] <= budget for st in r['stages'])
        elif a.mode == 'plan':
            r = plan(model, budget) or {'infeasible': True}
        else:
            r = plan_memory_balanced(model, budget, recompute=a.recompute == 'all') or {'infeasible': True}
        r = {'b': b, 'm': model.m, **r}
        results.append(r)
        if 'stages' in r:
            show(f'b={b} m={model.m}', r)
            if not r.get('fits', True):
                over = [s for s, st in enumerate(r['stages']) if st['reserve'] > budget]
                print(f'b={b}: DOES NOT FIT, stages {over} need more than the budget')
        else:
            print(f'b={b}: no split fits')
    fit = [r for r in results if 'stages' in r]
    keys = ('b', 'm', 'parts', 'layers_per_stage', 'activation_checkpoints_per_stage', 'step_ms')
    if a.mode == 'plan':
        # the smallest step time is the highest throughput, because the global batch is the same for every size;
        # the sizes are in ascending order, so the smallest size wins among equal step times
        best = min(fit, key=lambda r: r['step_ms'], default=None)
        print('BEST ' + (json.dumps({k: best[k] for k in keys}) if best else 'none'))
        for name, ckpts in (('no block recomputed', 'none'), ('every block recomputed', 'all')) if best else ():
            model = models[best['b']]
            r = given(model, even_parts(model), ckpts)
            over = [s for s, st in enumerate(r['stages']) if st['reserve'] > budget]
            if over:
                print(f"EVEN SPLIT, {name}: stages {over} do not fit")
            else:
                print(f"EVEN SPLIT, {name}: step {r['step_ms']:.0f} ms, throughput "
                      f"{100 * (best['step_ms'] / r['step_ms'] - 1):+.1f} % against the plan")
    elif a.mode == 'balance-memory':
        # the largest micro-batch size that fits uses the most of the device
        best = max(fit, key=lambda r: r['b'], default=None)
        print('BEST ' + (json.dumps({k: best[k] for k in keys}) if best else 'none'))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(results, f, indent=1, default=float)


if __name__ == '__main__':
    main()
