# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Two-level expert dispatch for AutoEP's collective path.

A token routed to k experts is k rows on the flat all-to-all, even when several of those experts sit on one
node and the rows are identical. Here a token crosses the network once per destination node: one row, the
pilot, goes over the expert-parallel group to the rank of one of the token's experts in that node, and that rank
copies it to the token's other experts in the node over a node-local all-to-all. On the way back the rank that
received the pilot adds the node's weighted expert outputs into one row, so the return trip also crosses the
network once per node.

The experts see the same rows as on the flat path, grouped by expert, so this changes only how rows move. One
count exchange gives every split size, and the split sizes are the only numbers the host reads from the device, so
the host waits for the device once per layer, as on the flat path. The choice of pilot is a function of the token
and the expert, so a checkpoint recompute repeats it. Both sums, on the node and per token, run in float32 and round
once; on a GPU they add in a fixed order, so two runs give the same bits there.

Collectives per layer: 7 in the forward (counts, the copies' addresses, the weights, the pilots, the copies out and
back inside the node, the return) and 5 in the backward, against 3 and 2 on the flat path. The extra ones carry
integers, weights or node-local rows; what they buy is fewer rows on the network.
"""

from __future__ import annotations

import socket
from typing import Callable, NamedTuple

import torch
import deepspeed.comm as dist


class TwoLevelTopology(NamedTuple):
    rank_node: torch.Tensor  # [ep_size] the node index of each EP rank
    rank_local: torch.Tensor  # [ep_size] each EP rank's index inside its node
    local_group: object  # this rank's node group, None when the node holds one EP rank
    local_size: int  # EP ranks in this rank's node
    max_local_size: int  # EP ranks in the largest node of this EP group
    num_nodes: int  # nodes this EP group spans


_TOPOLOGIES: dict = {}


def two_level_topology(ep_group, ep_group_name: str) -> TwoLevelTopology:
    """Build, once per EP layout, the node groups of every EP group. A node is a host name.

    Group creation is collective over the whole job, so every rank creates every EP group's node groups in the
    same order.
    """
    key = (ep_group_name, tuple(dist.get_all_ranks_from_group(ep_group)))
    if key in _TOPOLOGIES:
        return _TOPOLOGIES[key]

    table = [None] * dist.get_world_size()
    dist.all_gather_object(table, (tuple(dist.get_all_ranks_from_group(ep_group)), socket.gethostname()))
    host_of_rank = {world_rank: host for world_rank, (_, host) in enumerate(table)}
    ep_tables = sorted({ep_ranks for ep_ranks, _ in table})

    my_rank = dist.get_rank()
    topology = None
    for ep_ranks in ep_tables:
        node_of = {}
        members: list[list[int]] = []
        rank_node, rank_local = [], []
        for ep_index, world_rank in enumerate(ep_ranks):
            node_key = host_of_rank[world_rank]
            if node_key not in node_of:
                node_of[node_key] = len(members)
                members.append([])
            node = node_of[node_key]
            rank_node.append(node)
            rank_local.append(len(members[node]))
            members[node].append(world_rank)
        local_group, local_size = None, 1
        for ranks in members:
            group = dist.new_group(ranks) if len(ranks) > 1 else None
            if my_rank in ranks:
                local_group, local_size = group, len(ranks)
        if my_rank in ep_ranks:
            topology = TwoLevelTopology(torch.tensor(rank_node, dtype=torch.long),
                                        torch.tensor(rank_local, dtype=torch.long), local_group, local_size,
                                        max(len(ranks) for ranks in members), len(members))
    _TOPOLOGIES[key] = topology
    return topology


def _exchange_rows(rows: torch.Tensor, group, input_splits: list[int], output_splits: list[int]) -> torch.Tensor:
    """All-to-all of metadata rows, outside autograd."""
    received = rows.new_empty((sum(output_splits), *rows.shape[1:]))
    dist.all_to_all_single(received,
                           rows.contiguous(),
                           output_split_sizes=output_splits,
                           input_split_sizes=input_splits,
                           group=group)
    return received


class _SumRowsByToken(torch.autograd.Function):
    """Add ``rows`` [N, H] into ``num_tokens`` rows by ``index``, in float32 or wider, and round once.

    On CUDA devices ``index_put_`` with ``accumulate`` sorts by index and adds the rows of one token in a fixed
    order, which ``index_add_`` does not, so two runs give the same bits there. On the CPU the order of the additions
    can differ between runs. The graph keeps the index only.
    """

    @staticmethod
    def forward(ctx, rows, index, num_tokens):
        ctx.save_for_backward(index)
        total = rows.new_zeros((num_tokens, rows.shape[1]), dtype=torch.promote_types(rows.dtype, torch.float32))
        total.index_put_((index, ), rows.to(total.dtype), accumulate=True)
        return total.to(rows.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        index, = ctx.saved_tensors
        return grad_output.index_select(0, index), None, None


class _WeightedNodeSum(torch.autograd.Function):
    """``pilot_weight * pilot`` plus, row by row of ``copy_row``, ``copy_weight * copy``: what one node returns.

    The products and the sum run in float32 or wider and round once, in the order of ``_SumRowsByToken``. Without
    weights (scores applied before the experts) it is the plain sum, and the graph keeps ``copy_row`` only.
    """

    @staticmethod
    def forward(ctx, pilot, pilot_weight, copy, copy_weight, copy_row):
        wide = torch.promote_types(pilot.dtype, torch.float32)
        ctx.weighted = pilot_weight is not None
        if ctx.weighted:
            ctx.save_for_backward(pilot, pilot_weight, copy, copy_weight, copy_row)
            total, added = pilot * pilot_weight.to(wide), copy * copy_weight.to(wide)
        else:
            ctx.save_for_backward(copy_row)
            total, added = pilot.to(wide, copy=True), copy.to(wide)
        total.index_put_((copy_row, ), added, accumulate=True)
        return total.to(pilot.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.weighted:
            copy_row, = ctx.saved_tensors
            return grad_output, None, grad_output.index_select(0, copy_row), None, None
        pilot, pilot_weight, copy, copy_weight, copy_row = ctx.saved_tensors
        grad = grad_output.to(torch.promote_types(grad_output.dtype, torch.float32))
        grad_copy = grad.index_select(0, copy_row)
        return ((grad * pilot_weight).to(pilot.dtype), (grad * pilot).sum(1, keepdim=True).to(pilot_weight.dtype),
                (grad_copy * copy_weight).to(copy.dtype),
                (grad_copy * copy).sum(1, keepdim=True).to(copy_weight.dtype), None)


def _interleaved(first: torch.Tensor, second: torch.Tensor, num_first: int,
                 num_second: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Where two per-rank blocks sit when laid out rank by rank, a rank's first block before its second.

    ``first`` and ``second`` are the block sizes per rank, on the device; their sums are passed in so that nothing
    is read back from the device."""
    in_first = torch.arange(num_first, device=first.device) + torch.repeat_interleave(
        torch.cumsum(second, 0) - second, first, output_size=num_first)
    in_second = torch.arange(num_second, device=first.device) + torch.repeat_interleave(
        torch.cumsum(first, 0), second, output_size=num_second)
    return in_first, in_second


def _choose_pilots(selected_experts: torch.Tensor, dest_node: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each token and destination node, pick the slot whose row crosses the network.

    Returns ``order`` [T, K], the slots sorted by node, and ``pilot_pos`` [T, K], the position in ``order`` of the
    pilot of each sorted slot. A hash of the token and the expert spreads the pilots over the ranks of a node.
    """
    token_ids = torch.arange(selected_experts.shape[0], device=selected_experts.device).unsqueeze(1)
    mixed = (selected_experts.long() * 2654435761 + token_ids * 40503) & 0x7FFFFFFF
    order = torch.argsort((dest_node << 32) | mixed, dim=1, stable=True)
    sorted_node = dest_node.gather(1, order)
    first_in_node = torch.ones_like(sorted_node, dtype=torch.bool)
    first_in_node[:, 1:] = sorted_node[:, 1:] != sorted_node[:, :-1]
    positions = torch.arange(order.shape[1], device=order.device).expand_as(order)
    pilot_pos = torch.cummax(positions * first_in_node, dim=1).values
    return order, pilot_pos


def two_level_route(tokens: torch.Tensor, top_scores: torch.Tensor, selected_experts: torch.Tensor, *,
                    topology: TwoLevelTopology, ep_group, ep_size: int, num_experts: int, num_local_experts: int,
                    score_apply: str, run_experts: Callable[[torch.Tensor, torch.Tensor],
                                                            torch.Tensor], all_to_all_v: Callable) -> torch.Tensor:
    """Move tokens [T, H] to their experts and back. Returns the weighted sum per token, [T, H]."""
    device = tokens.device
    num_tokens, top_k = selected_experts.shape
    local_experts, max_local = num_local_experts, topology.max_local_size
    rank_node, rank_local = topology.rank_node, topology.rank_local  # on the device (the layer moves them once)

    with torch.no_grad():
        # Slots as [T * K]: a token's slots side by side, grouped by destination node, each naming its pilot.
        experts = selected_experts.long()
        order, pilot_pos = _choose_pilots(experts, rank_node[experts // local_experts])
        experts = experts.gather(1, order).reshape(-1)
        slot = torch.arange(num_tokens * top_k, device=device)
        pilot_of = (slot.view(num_tokens, top_k)[:, :1] + pilot_pos).reshape(-1)
        is_copy = pilot_of != slot
        rank = experts // local_experts
        pilot_rank = rank[pilot_of]
        local_expert = experts % local_experts

        # One count exchange. A rank tells each pilot rank (a) its pilots by expert and (b) its copies by the rank
        # in the node they go on to, and tells each copy's rank (c) its copies by the pilot's rank in the node and
        # by expert. From these every rank knows all six split lists and its experts' row counts.
        width = local_experts + max_local + max_local * local_experts
        sent = torch.zeros(ep_size * width, dtype=torch.long, device=device)
        sent.scatter_add_(0, pilot_rank * width + torch.where(is_copy, local_experts + rank_local[rank], local_expert),
                          torch.ones_like(experts))
        sent.scatter_add_(
            0, rank * width + local_experts + max_local + rank_local[pilot_rank] * local_experts + local_expert,
            is_copy.long())
        received = torch.empty_like(sent)
        dist.all_to_all_single(received, sent, group=ep_group)
        sent, received = sent.view(ep_size, width), received.view(ep_size, width)
        pilot_counts = received[:, :local_experts]  # [source rank, local expert]
        onward = slice(local_experts, local_experts + max_local)
        local_counts = received[:, onward.stop:].view(ep_size, max_local, local_experts).sum(0)
        local_counts = local_counts[:topology.local_size]  # [source rank in the node, local expert]
        split_sizes = torch.cat(
            (sent[:, :local_experts].sum(1), pilot_counts.sum(1), sent[:, onward].sum(1), received[:, onward].sum(1),
             received[:, onward].sum(0)[:topology.local_size], local_counts.sum(1)))
        # The one transfer to the host: the collectives take their split sizes as Python lists. What stays on the
        # device below is computed from the device copy, so the host does not wait a second time.
        splits = split_sizes.cpu().tolist()
        pilot_in, pilot_out, copy_in, copy_out = (splits[i * ep_size:(i + 1) * ep_size] for i in range(4))
        local_in = splits[4 * ep_size:4 * ep_size + topology.local_size]
        local_out = splits[4 * ep_size + topology.local_size:]
        pilots_sent, pilots_received, copies_sent, copies_received = split_sizes[:4 * ep_size].view(4, ep_size)

        # Send order: the pilots by expert, then the copies by their pilot's rank.
        send_order = torch.argsort(torch.where(is_copy, num_experts + pilot_rank, experts), stable=True)
        pilot_index, copy_index = send_order[:sum(pilot_in)], send_order[sum(pilot_in):]
        pilot_token = pilot_index // top_k
        sent_at = torch.empty_like(send_order)
        sent_at[send_order] = slot

        # A copy's address: its pilot's row within the block the pilot's rank receives from this rank, and its expert.
        block_start = pilots_sent.cumsum(0) - pilots_sent
        address = torch.stack(
            (sent_at[pilot_of[copy_index]] - block_start[pilot_rank[copy_index]], experts[copy_index]), dim=1)
        address = _exchange_rows(address, ep_group, copy_in, copy_out)

        # On the pilot's rank: the received row each copy reads, in the order the node's ranks and experts need.
        source = torch.repeat_interleave(torch.arange(ep_size, device=device),
                                         copies_received,
                                         output_size=sum(copy_out))
        by_expert = torch.argsort(address[:, 1], stable=True)
        copy_row = (address[:, 0] + (pilots_received.cumsum(0) - pilots_received)[source])[by_expert]

        # The weights of a rank's pilots and copies travel together, pilots first.
        pilots_at, copies_at = _interleaved(pilots_sent, copies_sent, sum(pilot_in), sum(copy_in))
        weight_order = torch.empty_like(send_order)
        weight_order[pilots_at], weight_order[copies_at] = pilot_index, copy_index
        pilots_at, copies_at = _interleaved(pilots_received, copies_received, sum(pilot_out), sum(copy_out))
        copies_at = copies_at[by_expert]

    def over_node(rows: torch.Tensor, input_splits: list[int], output_splits: list[int]) -> torch.Tensor:
        if topology.local_group is None:
            return rows
        return all_to_all_v(topology.local_group, rows, input_splits, output_splits)

    # The weights travel to the pilot's rank because the node's rows are added there.
    weights = top_scores.float().gather(1, order).reshape(-1, 1)
    weights = all_to_all_v(ep_group, weights[weight_order], [p + c for p, c in zip(pilot_in, copy_in)],
                           [p + c for p, c in zip(pilot_out, copy_out)])
    pilot_weight, copy_weight = weights[pilots_at], weights[copies_at]

    pilot_rows = all_to_all_v(ep_group, tokens[pilot_token], pilot_in, pilot_out)
    copy_rows = pilot_rows[copy_row]
    if score_apply == "pre":
        pilot_rows = (pilot_rows.float() * pilot_weight).to(tokens.dtype)
        copy_rows = (copy_rows.float() * copy_weight).to(tokens.dtype)
    copy_rows = over_node(copy_rows, local_in, local_out)

    expert_output = run_experts(torch.cat((pilot_rows, copy_rows)), torch.cat((pilot_counts, local_counts)))
    pilot_output, copy_output = expert_output.split([pilot_rows.shape[0], copy_rows.shape[0]])
    copy_output = over_node(copy_output, local_out, local_in)

    if score_apply == "post":
        if topology.local_group is not None:
            # The sum keeps its inputs for the backward, and a slice would keep the rows of other ranks' copies too.
            pilot_output = pilot_output.clone()
        node_output = _WeightedNodeSum.apply(pilot_output, pilot_weight, copy_output, copy_weight, copy_row)
    else:
        node_output = _WeightedNodeSum.apply(pilot_output, None, copy_output, None, copy_row)
    returned = all_to_all_v(ep_group, node_output, pilot_out, pilot_in)
    return _SumRowsByToken.apply(returned, pilot_token, num_tokens)
