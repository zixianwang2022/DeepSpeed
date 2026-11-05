# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""The memory of one pipeline stage, in bytes, at the three peaks of a training step.

Memory = parameters + gradients + optimizer state + the activations of the micro-batches in flight + the backward's
working set + buffers. Every term is either computed from DeepSpeed's rules (ZeRO stage 1 in
``runtime/zero/stage_1_and_2.py``, bf16 parameters, gradients and communication, the 1F1B schedule of
``runtime/pipe``) or measured per layer type by ``profile.py``; none is fitted to a run. Where the code allows a range,
the term takes the upper end, so a stage is predicted to need at least what it uses. A fixed headroom for what lies
outside PyTorch's allocator is subtracted from the device memory by the caller.

Terms of a stage:
  params, grads  2 bytes per parameter of the stage's layers (the local shard under tensor parallelism); the layers'
                 registered buffers are added to params
  opt            12 bytes per element of the rank's ZeRO-1 partition of every optimizer group (fp32 master, two moments)
  B              one reduce bucket: reduce_bucket_size x 2 bytes. The gradient hooks allocate one at the first backward
                 and the pipeline's reduce_gradients appends a second (``buckets`` = 2)
  G_part         the rank's gradient partition: 2 bytes x the whole parameters that overlap the rank's partition
  oversize       2 bytes x the largest parameter with more elements than reduce_bucket_size: such a gradient stays
                 out of the bucket and the reduce flattens it into a copy (allreduce_bucket)
  step           what the optimizer step holds for one group at a time, ``step_bytes`` x the largest group partition:
                 6 for the flattened bf16 gradient and its fp32 copy, 8 for the fp64 copy of a gradient that the
                 gradient norm makes, 12 for the fp32 gradient and the two temporaries of torch.optim's Adam and AdamW
                 (the default, 12, covers them all; an optimizer that updates in place needs 8)
  W              the GEMM workspaces (measured once)
  pipe           the receive buffer of the activation (stages after the first) and of the gradient (stages before the
                 last)
  kept, transient, stale, out, forward extra   per layer type, measured: what a micro-batch keeps until its backward,
                 the backward's peak, what stays after it, the output, and the forward's peak above what it keeps
  data           the micro-batch's inputs and labels on the device (``input_bytes``, ``label_bytes``; token ids are
                 negligible). The first stage keeps the input of each of its buffers, with its gradient when it is
                 floating point (counted always), and a copy per micro-batch in flight; the last stage keeps the labels
                 of each buffer
  held           what the last stage keeps after a backward: it does not release the buffer (engine.py,
                 _exec_backward_pass returns early), so per buffer the stage's output stays and, through the output's
                 graph, the received activation and its gradient, until the buffer is used again; and the engine
                 keeps the loss of every micro-batch of the step (the storage the loss tensor lives in, measured)
  slack          1 MiB per tensor of at least 1 MiB: the allocator hands out a cached block whole when under 1 MiB of
                 it would be left over (kSmallSize in c10/cuda/CUDACachingAllocator.cpp), and counts it as allocated
n_s = min(P - s, m) micro-batches are in flight on stage s of P under 1F1B, in max(2, n_s) buffers (pipe/schedule.py).
Peaks, with base = params + opt + W + pipe + data:
  1F1B    base + grads + B + n_s (2 x input + sum kept) + input (the gradient that a backward leaves on the received
          activation until it is sent) + held by the buffers not in flight + slack(gradients, n_s x kept)
          + max(max_i forward extra_i, max_i (transient_i + out_i + out_last + sum_{j > i} (stale_j - kept_j)))
  reduce  base + grads + sum stale + held + buckets x B + G_part + oversize + slack(gradients)
  step    base + sum stale + held + G_part + max(B, step)

What a stage needs from the device (``reserve``) is more than the largest of the three peaks. PyTorch's allocator
keeps a freed block and cuts it for later, smaller requests; it cannot move a live tensor. The reduce and the step
phases each allocate a few large tensors (the second reduce bucket, G_part, the copies of one optimizer group) that the
1F1B phase does not hold. If the 1F1B phase grows into the space those tensors left, the parameter gradients, which are
allocated in that phase and are still alive in the reduce, pin the pieces, and the next reduce finds no block of the
size of G_part although as much memory is free. So the three phases are laid side by side:
  reserve = max(1F1B, base + grads + B + sum stale + held + slack(gradients))
            + (buckets - 1) x B + G_part + oversize + step
The search compares ``reserve`` with the budget; ``peak``, the largest of the three peaks, is the estimate of
``max_memory_allocated``. A run with free device memory can show a larger ``max_memory_reserved`` than ``reserve``,
because the allocator returns cached blocks to the device only when a request fails.

Not covered: fp32 communication (``communication_data_type: fp32``, or sequence parallelism, which sets it), gradient
accumulation in fp32 (``BF16_Optimizer``), ZeRO stages 2 and 3, and CPU offload.
"""

GiB = 2**30
ZERO_ALIGN = 2  # stage_1_and_2.py: nccl_start_alignment_factor
SLACK = 2**20  # CUDACachingAllocator.cpp: kSmallSize, the most a block handed out whole exceeds the request by


def data_parallel_sizes(world_size, pp, tp=1, ep=1, expert_tensor_parallel=False):
    """The data-parallel size of the dense parameters and of the expert parameters, as DeepSpeed builds the groups.

    Dense data parallel = GPUs / (PP x TP). The expert groups depend on the MoE layer's builder (moe/layer.py,
    MoE._create_process_groups):
    - without expert tensor parallelism (the default), _create_expert_and_data_parallel cuts EP groups out of the
      consecutive ranks of a stage, whatever TP is: the stage's ranks must be a multiple of EP, and the experts' data
      parallel = GPUs / (PP x EP);
    - with it, _create_expert_data_and_model_parallel cuts them out of the data-parallel ranks: the dense data-parallel
      size must be a multiple of EP, and the experts' data parallel = dense data parallel / EP.
    """
    if world_size % (pp * tp):
        raise ValueError(f'{world_size} GPUs cannot hold PP {pp} x TP {tp}')
    dp = world_size // (pp * tp)
    if expert_tensor_parallel:
        if dp % ep:
            raise ValueError(
                f'expert parallel {ep} with expert tensor parallelism needs a multiple of {ep} data-parallel '
                f'ranks; PP {pp} x TP {tp} on {world_size} GPUs leaves {dp}')
        return dp, dp // ep
    if (world_size // pp) % ep:
        raise ValueError(
            f'expert parallel {ep} needs a multiple of {ep} ranks per stage; {world_size} GPUs over PP {pp} '
            f'give {world_size // pp}')
    return dp, world_size // (pp * ep)


def optimizer_groups(dense, expert, max_group_size):
    """``split_params_into_different_moe_groups_for_optimizer`` (deepspeed/moe/utils.py): one dense group, and the expert
    parameters cut greedily, in order, into groups of at most ``max_group_size`` elements."""
    groups = [('dense', list(dense))]
    cur, size = [], 0
    for n in expert:
        if max_group_size is None or size + n <= max_group_size:
            cur.append(n)
            size += n
        else:
            groups.append(('expert', cur))
            cur, size = [n], n
    if cur:
        groups.append(('expert', cur))
    return [(kind, nums) for kind, nums in groups if nums]


def zero1_partition(numels, dp):
    """ZeRO-1's partition of one flat group over ``dp`` ranks: the group padded to a multiple of ZERO_ALIGN x dp and cut
    into dp equal parts. Returns the partition size and the largest sum of whole parameters that overlap one partition
    (what ``copy_grads_in_partition`` holds)."""
    flat = sum(numels)
    padded = -(-flat // (ZERO_ALIGN * dp)) * ZERO_ALIGN * dp
    ps = padded // dp
    bounds, start = [], 0
    for n in numels:
        bounds.append((start, start + n))
        start += n
    worst = max(sum(e - s for s, e in bounds if s < (r + 1) * ps and e > r * ps) for r in range(dp))
    return ps, worst


class StageModel:
    """Memory and time of the stages of one model under one layout.

    costs     the costs file of ``profile.py`` (``load_costs``)
    layers    the layer type of every layer of the PipelineModule, in order
    blocks    the layer types that ``layers_per_stage`` counts and that may be recomputed (the partition type)
    pp, m, b  pipeline stages, micro-batches per step, micro-batch size
    dp, dp_e  data-parallel size of the dense and of the expert parameters (``data_parallel_sizes``)
    doubled_bucket  the pipeline's reduce_gradients appends a second bucket (DeepSpeed without the single-bucket fix)
    step_bytes      bytes per element of the largest group partition that the optimizer step holds at once (see above)
    input_bytes, label_bytes  what one micro-batch's inputs and labels hold on the device (see ``data`` above)
    overhead_ms      time that every stage adds to each micro-batch outside its layers (the engine, the sends and
                     receives between stages); 55 ms is the value X-MoE-4D's planner uses
    optimizer_ratio  the optimizer step as a share of the forward and backward passes of a step; 0.10 is the value
                     X-MoE-4D's planner uses
    """

    def __init__(self,
                 costs,
                 layers,
                 blocks,
                 pp,
                 dp,
                 dp_e,
                 m,
                 b,
                 bucket_elems=5e8,
                 max_group_size=178956971,
                 doubled_bucket=True,
                 step_bytes=12,
                 input_bytes=0,
                 label_bytes=0,
                 overhead_ms=55.0,
                 optimizer_ratio=0.10):
        self.c, self.layers, self.blocks = costs['layers'], list(layers), set(blocks)
        self.pp, self.dp, self.dp_e, self.m, self.b = pp, dp, dp_e, m, b
        self.B = int(bucket_elems) * 2
        self.W = costs.get('blas_workspace_bytes', 0)
        self.max_group_size = max_group_size
        self.buckets = 2 if doubled_bucket else 1
        self.step_bytes = step_bytes
        self.bucket_elems = int(bucket_elems)
        self.input_bytes, self.label_bytes = input_bytes, label_bytes
        self.overhead_ms, self.optimizer_ratio = overhead_ms, optimizer_ratio

    def r(self, kind):
        return self.c[kind]['by_mbs'][str(self.b)]

    def cost(self, kind, field, ckpt=False):
        r = self.r(kind)
        for key in ((f'{field}_ckpt_ms', f'{field}_ckpt_bytes') if ckpt else ()) + (f'{field}_ms', f'{field}_bytes'):
            if key in r:
                v = r[key]
                return v['median'] if isinstance(v, dict) else v
        raise KeyError(f'{kind}: no {field} for micro-batch {self.b}')

    def states(self, kinds):
        dense = [n for k in kinds for _, n, ex in self.c[k]['param_list'] if not ex]
        expert = [n for k in kinds for _, n, ex in self.c[k]['param_list'] if ex]
        parts = [
            zero1_partition(nums, self.dp if g == 'dense' else self.dp_e)
            for g, nums in optimizer_groups(dense, expert, self.max_group_size)
        ]
        D, E = sum(dense), sum(expert)
        tensors = sum(2 * n >= SLACK for n in dense + expert)
        ps_sum = sum(ps for ps, _ in parts)
        return {
            'params': 2 * (D + E) + sum(self.c[k].get('buffer_bytes', 0) for k in kinds),
            'grads': 2 * (D + E),
            'opt': 12 * ps_sum,
            'master': 4 * ps_sum,
            'G_part': 2 * sum(w for _, w in parts),
            'oversize': 2 * max((n for n in dense + expert if n > self.bucket_elems), default=0),
            'step_group': self.step_bytes * max((ps for ps, _ in parts), default=0),
            'slack': SLACK * tensors,
            'groups': len(parts)
        }

    def stage(self, s, lo, hi, n_ckpt):
        """layers lo..hi-1 on stage s, the first n_ckpt blocks recomputed"""
        kinds = self.layers[lo:hi]
        ck, seen = [], 0
        for k in kinds:
            ck.append(k in self.blocks and seen < n_ckpt)
            seen += k in self.blocks
        st = self.states(kinds)
        first, last = s == 0, s == self.pp - 1
        n = min(self.pp - s, self.m)
        buffers = max(2, n)
        inp = 0 if first else self.r(self.layers[lo - 1])['out_bytes']
        kept = [self.cost(k, 'kept', rc) for k, rc in zip(kinds, ck)]
        trans = [self.cost(k, 'transient', rc) for k, rc in zip(kinds, ck)]
        fwd_extra = [self.cost(k, 'fwd_extra', rc) for k, rc in zip(kinds, ck)]
        outb = [self.r(k)['out_bytes'] for k in kinds]
        pipe = (0 if first else inp) + (0 if last else outb[-1])
        data = ((2 * buffers + n) * self.input_bytes if first else 0) + (buffers * self.label_bytes if last else 0)
        # what one buffer of the last stage keeps after its backward, and the losses of the step's micro-batches
        per_buffer = outb[-1] + 2 * inp if last else 0
        losses = self.m * self.r(kinds[-1]).get('loss_bytes', 0) if last else 0
        held = buffers * per_buffer + losses
        # a recomputed block's stale graph also holds its input, which the profiler cannot see (there it is the caller's)
        ins = [inp] + outb[:-1]
        stale = [self.cost(k, 'stale', rc) + (ins[j] if rc else 0) for j, (k, rc) in enumerate(zip(kinds, ck))]
        # the backward at layer i: its own peak, its output, the stage's output (the engine holds it until the backward
        # ends), and what the layers after it left behind in place of what they kept
        peak_bwd = max(trans[i] + (outb[i] + outb[-1] if i < len(kinds) - 1 else 0) +
                       sum(stale[j] - kept[j] for j in range(i + 1, len(kinds))) for i in range(len(kinds)))
        slack_kept = SLACK * n * sum(
            self.r(k)['kept_ckpt_tensors' if rc else 'kept_tensors'] for k, rc in zip(kinds, ck))
        base = st['params'] + st['opt'] + self.W + pipe + data
        mem = {
            'after_init':
            st['params'] + st['master'],
            '1f1b':
            base + st['grads'] + self.B + n * (2 * inp + sum(kept)) + inp + (buffers - n) * per_buffer + losses +
            max(peak_bwd, max(fwd_extra)) + st['slack'] + slack_kept,
            'reduce':
            base + st['grads'] + sum(stale) + held + self.buckets * self.B + st['G_part'] + st['oversize'] +
            st['slack'],
            'step':
            base + sum(stale) + held + st['G_part'] + max(self.B, st['step_group']),
        }
        # the three phases side by side: the 1F1B phase never grows into the space of the large tensors of the other two
        after_1f1b = base + st['grads'] + self.B + sum(stale) + held + st['slack']
        reserve = max(mem['1f1b'],
                      after_1f1b) + (self.buckets - 1) * self.B + st['G_part'] + st['oversize'] + st['step_group']
        t = sum(self.cost(k, 'fwd', rc) + self.cost(k, 'bwd', rc) for k, rc in zip(kinds, ck))
        return {
            'layers': hi - lo,
            'blocks': sum(k in self.blocks for k in kinds),
            'ckpt': n_ckpt,
            'time_ms': t,
            'mem': mem,
            'peak': max(mem['1f1b'], mem['reduce'], mem['step']),
            'reserve': reserve,
            'groups': st['groups']
        }

    def pass_ms(self, stages, padded=False):
        """The forward and backward passes of one step under 1F1B, as in X-MoE-4D's ELM-PP planner: the first
        micro-batch crosses every stage once, and each of the other m - 1 follows after the time of the slowest
        stage. Every stage adds ``overhead_ms`` to each micro-batch. ``padded`` takes every stage to be as slow as
        the slowest, (m + pp - 1) times its time: what ELM-PP reports for a memory-balanced split."""
        ts = [st['time_ms'] + self.overhead_ms for st in stages]
        return (self.m + len(ts) - 1) * max(ts) if padded else sum(ts) + (self.m - 1) * max(ts)

    def step_ms(self, stages, padded=False):
        """``pass_ms`` plus the optimizer step, taken as ``optimizer_ratio`` of it."""
        return self.pass_ms(stages, padded) * (1 + self.optimizer_ratio)

    def bubble(self, stages, padded=False):
        """The share of ``pass_ms`` in which the slowest stage has no micro-batch to work on."""
        total = self.pass_ms(stages, padded)
        busy = self.m * (max(st['time_ms'] for st in stages) + self.overhead_ms)
        return 1 - busy / total if total > 0 else 0.0
