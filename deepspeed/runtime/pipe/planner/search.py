# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""The search of X-MoE-4D's ELM-PP planner, on the layer list of a ``PipelineModule``.

The rules are those of ELM-PP:

1. A stage gets the fewest recomputed blocks with which what it needs from the device (``reserve`` of
   ``StageModel.stage``) fits the budget. Its time is the time of its layers, each recomputed block taking its time
   with recompute.
2. ``plan``: the split is the one whose slowest stage is fastest, found by a dynamic program over the number of
   blocks that the first k stages hold.
3. The step time of a split follows from its stage times (``StageModel.step_ms``). Among several micro-batch sizes
   the caller takes the one with the smallest step time, which is the highest throughput for a fixed global batch;
   among equal ones the smallest size.
4. ``plan_memory_balanced``: the split whose largest stage memory is smallest when every block is recomputed, or
   none. It is ELM-PP's reference for what balancing memory alone gives. Its step time is reported as ELM-PP
   reports it, with every stage taken to be as slow as the slowest; the caller takes the largest micro-batch size
   for which such a split fits.

Two things follow from the layer list of a ``PipelineModule`` and are not in ELM-PP, whose stages hold blocks only
and may be empty. A stage holds at least one block, because a stage without layers has nothing to train. Cuts fall
right before a block, so a layer that is not a block (an embedding, a head) stays with its neighbour, as in
``PipelineModule(layers_per_stage=...)``, and its time counts in its stage.
"""

import math
from functools import lru_cache


def predict(model, parts, ckpts):
    """The stages and the step time of a given split (layer boundaries) and recompute counts."""
    stages = [model.stage(s, parts[s], parts[s + 1], ckpts[s]) for s in range(model.pp)]
    return stages, model.step_ms(stages)


def even_parts(model):
    """The layer boundaries of the even split: every stage holds the same number of blocks, and the first stages
    hold one more when the blocks do not divide."""
    at, L = [i for i, k in enumerate(model.layers) if k in model.blocks], len(model.layers)
    base, rem = divmod(len(at), model.pp)
    ends = [sum(base + (s < rem) for s in range(k)) for k in range(model.pp + 1)]
    return [0] + [at[i] for i in ends[1:-1]] + [L]


def _minimax(model, cost):
    """The split of the blocks into ``model.pp`` stages whose largest stage cost is smallest.

    ``cost(s, lo, hi, blocks)`` is a ``(value, stage)`` pair for stage s on layers lo..hi-1, or None when the stage
    does not fit. ``best[k][i]`` is the smallest possible largest cost of the first k stages when they hold the
    first i blocks; among equal values the earliest cut is kept. Returns the list of ``(lo, hi, stage)`` or None.
    """
    at, L, P = [i for i, k in enumerate(model.layers) if k in model.blocks], len(model.layers), model.pp
    N = len(at)
    if N < P:
        return None

    def span(j, i):
        return (0 if j == 0 else at[j]), (L if i == N else at[i])

    best = [[math.inf] * (N + 1) for _ in range(P + 1)]
    parent = [[None] * (N + 1) for _ in range(P + 1)]
    best[0][0] = 0.0
    for k in range(1, P + 1):
        for i in ([N] if k == P else range(k, N - (P - k) + 1)):
            for j in range(k - 1, i):
                if best[k - 1][j] == math.inf:
                    continue
                found = cost(k - 1, *span(j, i), i - j)
                if found is None:
                    continue
                largest = max(best[k - 1][j], found[0])
                if largest < best[k][i]:
                    best[k][i], parent[k][i] = largest, (j, found[1])
    if best[P][N] == math.inf:
        return None
    out, i = [], N
    for k in range(P, 0, -1):
        j, st = parent[k][i]
        out.append((*span(j, i), st))
        i = j
    return out[::-1]


def _result(model, found, padded=False):
    stages = [st for _, _, st in found]
    return {
        'parts': [lo for lo, _, _ in found] + [len(model.layers)],
        'layers_per_stage': [st['blocks'] for st in stages],
        'activation_checkpoints_per_stage': [st['ckpt'] for st in stages],
        'stages': stages,
        'step_ms': model.step_ms(stages, padded),
        'max_stage_ms': max(st['time_ms'] for st in stages),
        'bubble': model.bubble(stages, padded),
    }


def plan(model, budget_bytes):
    """The split whose slowest stage is fastest among those that fit ``budget_bytes`` on every stage, each stage
    with the fewest recomputed blocks that fit; None when no split fits."""

    @lru_cache(maxsize=None)
    def fastest(s, lo, hi, blocks):
        for c in range(blocks + 1):
            st = model.stage(s, lo, hi, c)
            if st['reserve'] <= budget_bytes:
                return st['time_ms'], st
        return None

    found = _minimax(model, fastest)
    return _result(model, found) if found else None


def plan_memory_balanced(model, budget_bytes, recompute=True):
    """The split whose largest ``reserve`` is smallest among those that fit ``budget_bytes`` on every stage, with
    every block recomputed (``recompute=True``) or none; None when no split fits."""

    def memory(s, lo, hi, blocks):
        st = model.stage(s, lo, hi, blocks if recompute else 0)
        return (st['reserve'], st) if st['reserve'] <= budget_bytes else None

    found = _minimax(model, memory)
    return _result(model, found, padded=True) if found else None
