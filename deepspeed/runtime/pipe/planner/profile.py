# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Measure each layer type of a pipeline model once, outside the training loop, and write the planner's costs file.

Nothing is added to the model or to the training: the profiler builds its own copy of each layer type, runs a forward
and a backward on it, and restores the random-number state it found. Per layer type and micro-batch size it records:
  fwd, bwd           forward and backward time, without and with recompute (``fwd_ckpt``, ``bwd_ckpt``), median and
                     the 10th and 90th percentiles of the timed runs
  kept               what one micro-batch keeps alive until its backward: the storages autograd saves, except the
                     parameters and the input, plus the output, and how many of them hold at least 1 MiB; on devices
                     with memory statistics, what the forward left allocated when that is more (it also sees what a
                     custom autograd function keeps on its context)
  loss               the storage that the loss tensor of a last layer lives in (the engine keeps every micro-batch's
                     loss until the next step, and a reduction can return a scalar inside a larger buffer)
  fwd_extra          the forward's peak above what is still allocated after it (devices with memory statistics)
  transient          the backward's extra peak above what was allocated before it (devices with memory statistics)
  stale              what the layer still holds after its backward (devices with memory statistics)
  out                the size of the output, which the pipeline sends to the next stage
and once, the parameter list of the layer type (name, number of elements, whether it is an expert parameter), the
bytes of its registered buffers, and the GEMM workspaces. A layer takes one tensor and returns one tensor. Under expert or tensor parallelism, build the layer in its real groups: the parameter counts and the
activations are then those of the local shard, and the numbers are the maximum over the ranks of ``group``.
"""

import hashlib
import json
import statistics
import time

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import is_moe_param
import deepspeed.comm as dist

SCHEMA = 4

MiB = 2**20


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def _memory_stats():
    acc = get_accelerator()
    try:
        return acc.memory_allocated() is not None and acc.device_name() != 'cpu'
    except Exception:
        return False


def _run_once(m, x, loss_fn, ckpt, checkpoint_fn, timer):
    """One forward and backward of layer m on input x. The parameter gradients are not cleared: after the first run they
    are accumulated in place, as for every micro-batch of a pipeline step after the first."""
    acc = get_accelerator()
    stats = _memory_stats()
    with torch.no_grad():  # replaces what the previous run left on the module, so that it is not counted as stale
        m(x)
    if x.requires_grad:
        x.grad = None
    acc.synchronize()
    a0 = acc.memory_allocated() if stats else 0
    exclude = {p.untyped_storage().data_ptr() for p in m.parameters()} | {x.untyped_storage().data_ptr()}
    saved = {}
    if stats:
        acc.reset_peak_memory_stats()

    def pack(t):
        st = t.untyped_storage()
        if st.data_ptr() not in exclude:
            saved[st.data_ptr()] = st.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        t0 = timer()
        out = checkpoint_fn(m, x) if ckpt else m(x)
        y = loss_fn(out) if loss_fn is not None else out
        t1 = timer()
    fwd_extra = (acc.max_memory_allocated() - acc.memory_allocated()) if stats else 0
    saved.setdefault(out.untyped_storage().data_ptr(), out.untyped_storage().nbytes())
    kept, kept_tensors = sum(saved.values()), sum(n >= MiB for n in saved.values())
    out_bytes = out.untyped_storage().nbytes()
    loss_bytes = y.untyped_storage().nbytes() if loss_fn is not None else 0
    if stats:
        kept = max(kept, acc.memory_allocated() - a0 - loss_bytes)
        acc.reset_peak_memory_stats()
    b0 = acc.memory_allocated() if stats else 0
    t2 = timer()
    y.backward(torch.ones_like(y) if y.dim() else None)
    t3 = timer()
    transient = (acc.max_memory_allocated() - b0) if stats else 0
    del out, y
    if x.requires_grad:
        x.grad = None
    acc.synchronize()
    stale = (acc.memory_allocated() - a0) if stats else 0
    return t1 - t0, t3 - t2, kept, transient, stale, out_bytes, fwd_extra, kept_tensors, loss_bytes


def profile_layer_types(factories,
                        make_input,
                        mbs_list,
                        reps=20,
                        warmup=5,
                        loss_fns=None,
                        checkpoint_fn=None,
                        recompute=None,
                        group=None):
    """Measure every layer type.

    factories     {layer type: a function that returns a new module of that type on the device}
    make_input    function (layer type, micro-batch size) -> the layer's input tensor
    loss_fns      {layer type: function(output) -> loss} for a last layer whose output is not what is backpropagated
    checkpoint_fn the activation-checkpoint function the run uses (default: deepspeed's checkpointing.checkpoint)
    recompute     the layer types that may be recomputed (default: all)
    group         the process group to take the maximum over (for example the expert-parallel group)
    Returns {layer type: {'param_list': [[name, numel, is_expert], ...], 'buffer_bytes': n, 'by_mbs': {str(b): {...}}}}.
    """
    acc = get_accelerator()
    if checkpoint_fn is None:
        from deepspeed.runtime.activation_checkpointing import checkpointing
        checkpoint_fn = checkpointing.checkpoint
    loss_fns = loss_fns or {}
    rng = torch.get_rng_state(), (acc.get_rng_state() if acc.device_name() != 'cpu' else None)

    def timer():
        acc.synchronize()
        return time.perf_counter() * 1e3

    rows = {}
    try:
        for kind, factory in factories.items():
            m = factory()
            rows[kind] = {
                'param_list': [[n, p.numel(), bool(is_moe_param(p))] for n, p in m.named_parameters()],
                'buffer_bytes':
                sum({b.untyped_storage().data_ptr(): b.untyped_storage().nbytes()
                     for b in m.buffers()}.values()),
                'by_mbs': {}
            }
            for b in mbs_list:
                res = {}
                for ckpt in ((False, True) if recompute is None or kind in recompute else (False, )):
                    x = make_input(kind, b)
                    runs = []
                    for r in range(warmup + reps):
                        if dist.is_initialized():
                            dist.barrier(group=group)
                        out = _run_once(m, x, loss_fns.get(kind), ckpt, checkpoint_fn, timer)
                        if r >= warmup:
                            runs.append(out)
                    fw, bw = [t[0] for t in runs], [t[1] for t in runs]
                    vals = [
                        statistics.median(fw),
                        _pct(fw, 0.1),
                        _pct(fw, 0.9),
                        statistics.median(bw),
                        _pct(bw, 0.1),
                        _pct(bw, 0.9)
                    ] + [float(max(t[i] for t in runs)) for i in range(2, 9)]
                    if dist.is_initialized() and group is not None:
                        v = torch.tensor(vals, dtype=torch.float64, device=acc.current_device_name())
                        dist.all_reduce(v, op=dist.ReduceOp.MAX, group=group)  # the slowest and the largest rank
                        vals = v.tolist()
                    sfx = '_ckpt' if ckpt else ''
                    res.update({
                        f'fwd{sfx}_ms': {
                            'median': vals[0],
                            'p10': vals[1],
                            'p90': vals[2]
                        },
                        f'bwd{sfx}_ms': {
                            'median': vals[3],
                            'p10': vals[4],
                            'p90': vals[5]
                        },
                        f'kept{sfx}_bytes': int(vals[6]),
                        f'transient{sfx}_bytes': int(vals[7]),
                        f'stale{sfx}_bytes': int(vals[8]),
                        f'fwd_extra{sfx}_bytes': int(vals[10]),
                        f'kept{sfx}_tensors': int(vals[11]),
                    })
                    res['out_bytes'] = int(vals[9])
                    res['loss_bytes'] = max(res.get('loss_bytes', 0), int(vals[12]))
                rows[kind]['by_mbs'][str(b)] = res
            del m
    finally:
        torch.set_rng_state(rng[0])
        if rng[1] is not None:
            acc.set_rng_state(rng[1])
    return rows


def write_costs(path, rows, key_fields, blas_workspace_bytes=0, extra=None):
    """Write the costs file. ``key_fields`` (device, versions, shapes, TP, EP, dtype) name what the costs are valid for;
    their hash is the file's key, and ``load_costs`` refuses a file whose fields do not match the run's."""
    key = hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()[:12]
    doc = {
        'schema': SCHEMA,
        'key': key,
        'key_fields': key_fields,
        'blas_workspace_bytes': blas_workspace_bytes,
        'layers': rows,
        **(extra or {})
    }
    with open(path, 'w') as f:
        json.dump(doc, f, indent=1)
    return key


def load_costs(path, expect=None):
    with open(path) as f:
        doc = json.load(f)
    if doc.get('schema') != SCHEMA:
        raise ValueError(f'{path}: costs schema {doc.get("schema")}, expected {SCHEMA}; profile again')
    if expect:
        # a parallel size that the file does not name counts as 1
        bad = {k: (doc['key_fields'].get(k, 1), v) for k, v in expect.items() if doc['key_fields'].get(k, 1) != v}
        if bad:
            raise ValueError(f'{path} was profiled for another layout (field: (file, run)): {bad}; profile again')
    return doc
