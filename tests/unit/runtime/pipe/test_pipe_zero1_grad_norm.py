# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import pytest
import torch
import torch.nn as nn

import deepspeed
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.runtime.pipe.topology import PipeDataParallelTopology
from unit.common import DistributedTest


class TestPipeZero1GradNormStages(DistributedTest):
    """The ZeRO-1 gradient norm is the norm of the whole model's gradient, however the layers are split into stages.

    Every Linear is its own optimizer parameter group, so the 3 + 1 split gives the stages 3 and 1 groups. Without one
    model-parallel reduction for all groups, the stage with more groups calls the cross-stage all-reduce more often than
    the other stage and the first optimizer step never ends.
    """
    world_size = 4

    @pytest.mark.parametrize("layers_per_stage", [[2, 2], [3, 1]])
    def test_norm_matches_unpartitioned_model(self, layers_per_stage):
        hidden, micro_batches, dp = 8, 4, 2
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": micro_batches,
            "zero_optimization": {
                "stage": 1
            },
            "zero_allow_untested_optimizer": True,
            "gradient_clipping": 1.0,
        }
        torch.manual_seed(0)
        layers = []
        for _ in range(4):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        reference = copy.deepcopy(nn.Sequential(*layers))
        loss_fn = nn.MSELoss()

        def batches(dp_id):
            g = torch.Generator().manual_seed(100 + dp_id)
            return [(torch.randn(1, hidden, generator=g), torch.randn(1, hidden, generator=g))
                    for _ in range(micro_batches)]

        model = PipelineModule(layers=layers,
                               topology=PipeDataParallelTopology(num_pp=2, num_dp=dp),
                               loss_fn=loss_fn,
                               partition_method='type:linear',
                               layers_per_stage=layers_per_stage)
        groups = [{"params": list(m.parameters())} for m in model.forward_funcs if isinstance(m, nn.Linear)]
        assert len(groups) == layers_per_stage[model.stage_id]
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, optimizer=torch.optim.SGD(groups, lr=0.0))
        engine.train_batch(data_iter=iter(batches(engine.grid.get_data_parallel_id())))

        # The gradient ZeRO-1 sees: the mean loss over every micro-batch of both data-parallel ranks.
        loss = sum(loss_fn(reference(x), y) for d in range(dp) for x, y in batches(d)) / (dp * micro_batches)
        loss.backward()
        expected = torch.sqrt(sum((p.grad.double()**2).sum() for p in reference.parameters()))
        assert torch.allclose(torch.tensor(float(engine.get_global_grad_norm()), dtype=torch.double),
                              expected,
                              rtol=1e-5)


class TestPipeZero1GradNormExpertGroups(DistributedTest):
    """The same with MoE layers, where the unequal group counts come from: the expert parameters are split into
    optimizer groups by ``max_group_size``, here one group per MoE layer, so the 3 + 1 split gives the stages 4 and 2
    groups. The norm must be the norm of the gradient partitions of every rank and group.
    """
    world_size = 4

    @pytest.mark.parametrize("layers_per_stage", [[2, 2], [3, 1]])
    def test_norm_matches_gradient_partitions(self, layers_per_stage):
        from deepspeed.moe.layer import MoE
        from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
        import deepspeed.comm as dist

        hidden, micro_batches, experts_per_rank = 8, 2, 2

        class Block(nn.Module):

            def __init__(self):
                super().__init__()
                self.dense = nn.Linear(hidden, hidden)
                self.moe = MoE(hidden_size=hidden,
                               expert=nn.Linear(hidden, hidden),
                               num_experts=2 * experts_per_rank,
                               ep_size=2,
                               k=1,
                               use_rts=False)

            def forward(self, x):
                return self.moe(torch.tanh(self.dense(x)))[0]

        config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": micro_batches,
            "zero_optimization": {
                "stage": 1
            },
            "zero_allow_untested_optimizer": True,
            "gradient_clipping": 1.0,
        }
        torch.manual_seed(0)
        model = PipelineModule(layers=[Block() for _ in range(4)],
                               topology=PipeDataParallelTopology(num_pp=2, num_dp=dist.get_world_size() // 2),
                               loss_fn=nn.MSELoss(),
                               partition_method='type:block',
                               layers_per_stage=layers_per_stage)
        one_layer = experts_per_rank * (hidden * hidden + hidden)
        groups = split_params_into_different_moe_groups_for_optimizer(
            {
                'params': list(model.parameters()),
                'name': 'all'
            }, max_group_size=one_layer)
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, optimizer=torch.optim.SGD(groups, lr=0.0))
        optimizer = engine.optimizer
        assert len(optimizer.bit16_groups) == 1 + layers_per_stage[model.stage_id]
        # the path under test: one reduction over the stages, and every expert held by world / (2 stages x EP 2) ranks
        from deepspeed.comm import pg_utils
        assert pg_utils.is_parallel(optimizer.model_parallel_group)
        expert_dp = dist.get_world_size() // 4
        assert [dist.get_world_size(group=optimizer.real_dp_process_group[i])
                for i in range(len(optimizer.bit16_groups)) if optimizer.is_moe_param_group[i]] == \
            [expert_dp] * layers_per_stage[model.stage_id]

        # the squared gradient partitions this rank holds when the norm is taken: over all ranks they are the gradient
        partition_squares = []
        scaled_global_norm = optimizer.scaled_global_norm

        def recording(*args, **kwargs):
            partition_squares.append(
                sum((g.double()**2).sum() for i in range(len(optimizer.bit16_groups))
                    for g in optimizer.averaged_gradients[i]))
            return scaled_global_norm(*args, **kwargs)

        optimizer.scaled_global_norm = recording
        g = torch.Generator().manual_seed(100 + engine.grid.get_data_parallel_id())
        batches = [(torch.randn(1, 16, hidden, generator=g), torch.randn(1, 16, hidden, generator=g))
                   for _ in range(micro_batches)]
        engine.train_batch(data_iter=iter(batches))

        assert len(partition_squares) == 1
        total = partition_squares[0].to(engine.device)
        dist.all_reduce(total)
        assert torch.allclose(torch.tensor(float(engine.get_global_grad_norm()), dtype=torch.double),
                              total.sqrt().cpu(),
                              rtol=1e-5)


class TestPipeZero1GradNormExpertDataParallel(TestPipeZero1GradNormExpertGroups):
    """The same on 8 ranks (2 stages x 4 data-parallel ranks, EP 2), so every expert has two data-parallel copies and
    its gradient norm also passes the expert data-parallel reduction. The expert-norm average that the per-group path
    applies, and that the once-per-step reduction leaves out, divides by 2 here: the norm must still be the norm of
    the gradient partitions of every rank and group.
    """
    world_size = 8
