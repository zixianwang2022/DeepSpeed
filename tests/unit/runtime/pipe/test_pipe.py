# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
import torch.nn as nn
import pytest

import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.runtime.pipe.topology import PipeDataParallelTopology
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.runtime.activation_checkpointing import checkpointing as ds_checkpointing
from deepspeed.utils import RepeatingLoader
from unit.alexnet_model import AlexNetPipe, train_cifar
from unit.common import DistributedTest
from unit.util import skip_on_arch, no_child_process_in_deepspeed_io

PipeTopo = PipeDataParallelTopology

config_dict = {
    "train_batch_size": 4,
    "grandient_accumulation_steps": 1,
    "steps_per_print": 20,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },
    "zero_optimization": {
        "stage": 0
    },
    "fp16": {
        "enabled": False
    },
    "pipeline": {
        "seed_layers": True,
        "activation_checkpoint_interval": 1
    }
}


def rel_diff(A, B):
    return abs(A - B) / abs(A)


class TestPipeGradientAccumulationScaling(DistributedTest):
    world_size = 2

    def test_gradients_are_scaled_once(self):
        gas = 2
        learning_rate = 0.1
        config = {
            "train_batch_size": gas,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": gas,
            "gradient_clipping": 0.0,
            "optimizer": {
                "type": "SGD",
                "params": {
                    "lr": learning_rate
                }
            },
            "zero_allow_untested_optimizer": True,
            "pipeline": {
                "activation_checkpoint_interval": 0
            },
        }

        layers = [nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)]
        for layer in layers:
            layer.weight.data.fill_(1.0)

        model = PipelineModule(layers=layers, num_stages=2, loss_fn=nn.MSELoss())
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        engine.set_dataiterator(RepeatingLoader([(torch.ones(1, 1), torch.zeros(1, 1))]))

        engine.train_batch()

        expected_weight = torch.tensor([[1.0 - 2.0 * learning_rate]], device=engine.device)
        actual_weight = next(engine.module.parameters())
        assert torch.allclose(actual_weight, expected_weight)


@pytest.mark.parametrize('topo_config', [
    {
        "num_pp": 1,
        "num_dp": 4
    },
    {
        "num_pp": 2,
        "num_dp": 2
    },
    {
        "num_pp": 4,
        "num_dp": 1
    },
])
class TestPipeCifar10(DistributedTest):
    world_size = 4

    def test_pipe_base(self, topo_config):
        skip_on_arch(min_arch=7)
        topo = PipeTopo(**topo_config)
        steps = 100  # must be >=100

        # Allocate model for consistent initial weights.
        init_net = AlexNetPipe()

        base_net = copy.deepcopy(init_net)
        base_model = PipelineModule(layers=base_net.to_layers(), num_stages=1, loss_fn=nn.CrossEntropyLoss())

        # Train with just data parallelism
        base_losses = train_cifar(base_model, config=config_dict, num_steps=steps, fp16=config_dict['fp16']['enabled'])

        test_net = copy.deepcopy(init_net)
        test_model = PipelineModule(layers=test_net.to_layers(), topology=topo, loss_fn=nn.CrossEntropyLoss())

        test_losses = train_cifar(test_model, config=config_dict, num_steps=steps, fp16=config_dict['fp16']['enabled'])

        abs_diffs = [l0 - l1 for l0, l1 in zip(base_losses, test_losses)]
        rel_diffs = [rel_diff(l0, l1) for l0, l1 in zip(base_losses, test_losses)]
        if dist.get_rank() == 0:
            print(f'abs min={min(abs_diffs)} max={max(abs_diffs)} avg={sum(abs_diffs)/len(abs_diffs)}')
            print(f'rel min={min(rel_diffs)} max={max(rel_diffs)} avg={sum(rel_diffs)/len(rel_diffs)}')
            print(f'first: base={base_losses[0]} test={test_losses[0]} abs={abs_diffs[0]} rel={rel_diffs[0]}')

            for lastX in [1, 10, 100]:
                base_avg = sum(base_losses[-lastX:]) / lastX
                test_avg = sum(test_losses[-lastX:]) / lastX
                print(
                    f'last-{lastX}: base={base_avg} test={test_avg} abs={base_avg - test_avg} rel={rel_diff(base_avg, test_avg)}'
                )

        lastX = 100
        base = base_losses[-lastX:]
        base_avg = sum(base) / len(base)
        test = test_losses[-lastX:]
        test_avg = sum(test) / len(test)
        assert rel_diff(base_avg, test_avg) < 0.05  # Originally 0.03, but seeing instability with AMD results

    # def _check_model_params_equal(self, model1, model2):
    #     for p1, p2 in zip(model1.parameters(), model2.parameters()):
    #         if p1.data.ne(p2.data).sum() > 0:
    #             assert False, f"model params not equal"

    def test_pipe_use_reentrant(self, topo_config):
        skip_on_arch(min_arch=7)

        topo = PipeTopo(**topo_config)
        steps = 100  # must be >=100

        # Allocate model for consistent initial weights.
        init_net = AlexNetPipe()

        # Train with not set use_reentrant, default: True
        base_net = copy.deepcopy(init_net)
        base_model = PipelineModule(layers=base_net.to_layers(), topology=topo, loss_fn=nn.CrossEntropyLoss())
        base_losses = train_cifar(base_model, config=config_dict, num_steps=steps, fp16=config_dict['fp16']['enabled'])

        # Train with set use_reentrant=False, this will use ``non_reentrant_checkpoint``
        test_config_dict = copy.deepcopy(config_dict)
        test_config_dict['pipeline']['use_reentrant'] = False
        test_net = copy.deepcopy(init_net)
        test_model = PipelineModule(layers=test_net.to_layers(), topology=topo, loss_fn=nn.CrossEntropyLoss())
        test_losses = train_cifar(test_model,
                                  config=test_config_dict,
                                  num_steps=steps,
                                  fp16=config_dict['fp16']['enabled'])

        abs_diffs = [l0 - l1 for l0, l1 in zip(base_losses, test_losses)]
        rel_diffs = [rel_diff(l0, l1) for l0, l1 in zip(base_losses, test_losses)]
        if dist.get_rank() == 0:
            print(f'abs min={min(abs_diffs)} max={max(abs_diffs)} avg={sum(abs_diffs)/len(abs_diffs)}')
            print(f'rel min={min(rel_diffs)} max={max(rel_diffs)} avg={sum(rel_diffs)/len(rel_diffs)}')
            print(f'first: base={base_losses[0]} test={test_losses[0]} abs={abs_diffs[0]} rel={rel_diffs[0]}')

            for lastX in [1, 10, 100]:
                base_avg = sum(base_losses[-lastX:]) / lastX
                test_avg = sum(test_losses[-lastX:]) / lastX
                print(
                    f'last-{lastX}: base={base_avg} test={test_avg} abs={base_avg - test_avg} rel={rel_diff(base_avg, test_avg)}'
                )
        lastX = 100
        base = base_losses[-lastX:]
        base_avg = sum(base) / len(base)
        test = test_losses[-lastX:]
        test_avg = sum(test) / len(test)
        assert rel_diff(base_avg, test_avg) < 0.05

        # the following check could passed on higher version docker: nvcr.io/nvidia/pytorch:23.07-py3(torch2.1.0 cuda12.1)
        # Check if models have same weights after training
        # self._check_model_params_equal(base_model, test_model)


class DynamicShapeTestLayer(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.shapes = set()

    def forward(self, x):
        self.shapes.add(x.shape)
        y = self.fc(x)
        return y


class DynamicShapeTestModel(nn.Module):

    def __init__(self, n_layers, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList([DynamicShapeTestLayer(hidden_size) for _ in range(n_layers)])


@pytest.mark.parametrize('topo_config', [
    {
        "num_pp": 1,
        "num_dp": 4
    },
    {
        "num_pp": 2,
        "num_dp": 2
    },
    {
        "num_pp": 4,
        "num_dp": 1
    },
])
class TestPipeDynamicShape(DistributedTest):
    world_size = 4

    def test_pipe_base(self, topo_config):
        """This test checks if the pipeline engine can handle dynamic shapes correctly.
        We pass inputs of different shapes to the pipeline engine.
        """

        n_iter = 10
        n_layers = 4
        n_samples = 1024
        batch_size = 4
        channel_dims = [8, 16, 32, 64]
        hidden_size = 16

        topo = PipeTopo(**topo_config)

        model = DynamicShapeTestModel(n_layers, hidden_size)
        model = PipelineModule(layers=model.layers, topology=topo, loss_fn=nn.MSELoss(), dynamic_shape=True)

        # Each batch has different channel dim but we use the same channel dim in the same batch
        xs = [
            torch.randn(channel_dims[(i // batch_size) % len(channel_dims)], hidden_size, dtype=torch.float32)
            for i in range(n_samples)
        ]
        ys = [torch.randn_like(x) for x in xs]

        class CustomDataset(torch.utils.data.Dataset):

            def __init__(self, xs, ys):
                self.xs = xs
                self.ys = ys

            def __len__(self):
                return len(self.xs)

            def __getitem__(self, idx):
                return self.xs[idx], self.ys[idx]

        dataset = CustomDataset(xs, ys)

        config_dict["train_batch_size"] = batch_size

        with no_child_process_in_deepspeed_io():
            engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                                   model=model,
                                                   model_parameters=[p for p in model.parameters()],
                                                   training_data=dataset)

        for _ in range(n_iter):
            _ = engine.train_batch()

        # Check if all layers have seen different shapes
        for layer in model.modules():
            if isinstance(layer, DynamicShapeTestLayer):
                assert len(layer.shapes) > 1


class TestDualPipeV(DistributedTest):
    world_size = [1, 2, 4]

    @pytest.mark.parametrize("checkpoint_interval", [0, 1])
    def test_matches_sequential_model(self, checkpoint_interval, tmpdir):
        """DualPipeV must give the same loss and weights as the unpartitioned model.
        A checkpoint saved by one engine must restore the same weights in a fresh engine.
        """
        from deepspeed.pipe import DualPipeVModule

        num_ranks = dist.get_world_size()
        micro_batches = 2 * num_ranks + 1
        hidden, lr = 8, 0.1
        config = {
            "train_batch_size": micro_batches,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": micro_batches,
            "optimizer": {
                "type": "SGD",
                "params": {
                    "lr": lr
                }
            },
            "zero_allow_untested_optimizer": True,
            "pipeline": {
                "activation_checkpoint_interval": checkpoint_interval
            },
        }
        torch.manual_seed(0)
        layers = []
        for _ in range(2 * num_ranks):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        reference = copy.deepcopy(nn.Sequential(*layers))
        data = [(torch.randn(1, hidden), torch.randn(1, hidden)) for _ in range(micro_batches)]
        loss_fn = nn.MSELoss()

        model = DualPipeVModule(layers=layers, num_stages=num_ranks, loss_fn=loss_fn, partition_method='uniform')
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        engine.set_dataiterator(RepeatingLoader(data))

        def reference_step():
            losses = []
            for x, y in data:
                loss = loss_fn(reference(x), y)
                (loss / micro_batches).backward()
                losses.append(loss.detach())
            with torch.no_grad():
                for p in reference.parameters():
                    p -= lr * p.grad
                    p.grad = None
            return torch.stack(losses).mean()

        def assert_nothing_retained():
            # Slots are per micro-batch, so anything left over is held until the next step overwrites it.
            for key in ('inputs', 'labels', 'outputs', 'grads', 'losses'):
                assert all(slot is None for slot in engine.pipe_buffers[key]), key

        reference_params = dict(reference.named_parameters())
        for _ in range(2):
            loss = engine.train_batch()
            assert torch.allclose(loss.cpu(), reference_step(), atol=1e-5)
            for name, p in engine.module.named_parameters():
                assert torch.allclose(p.detach().cpu(), reference_params[name], atol=1e-5), name
            assert_nothing_retained()

        # Too few micro-batches must be rejected before the eval iterator replaces the training one.
        with pytest.raises(ValueError):
            engine.eval_batch(iter(data), num_micro_batches=1)
        eval_loss = engine.eval_batch(iter(data))
        with torch.no_grad():
            expected = torch.stack([loss_fn(reference(x), y) for x, y in data]).mean()
        assert torch.allclose(eval_loss.cpu(), expected, atol=1e-5)
        assert_nothing_retained()
        loss = engine.train_batch()
        assert torch.allclose(loss.cpu(), reference_step(), atol=1e-5)

        engine.save_checkpoint(tmpdir)
        torch.manual_seed(1)
        fresh = DualPipeVModule(
            layers=[nn.Linear(hidden, hidden) if isinstance(l, nn.Linear) else nn.Tanh() for l in layers],
            num_stages=num_ranks,
            loss_fn=loss_fn,
            partition_method='uniform')
        fresh_engine, _, _, _ = deepspeed.initialize(config=config, model=fresh, model_parameters=fresh.parameters())
        fresh_engine.load_checkpoint(tmpdir)
        for name, p in fresh_engine.module.named_parameters():
            assert torch.allclose(p.detach().cpu(), reference_params[name], atol=1e-5), name


class TestPipeLayersPerStage(DistributedTest):
    world_size = 2

    # Six (Linear, Tanh) pairs; only the Linear layers count. DualPipeV cuts the model into twice as many stages as
    # there are ranks, and rank r runs stages r and 3 - r. With embedding_first, an embedding of token ids comes first:
    # it has parameters but is not counted, so it must not take the place of a counted block; checkpointed with
    # reentrant checkpointing, it would get no gradient (its input, the token ids, cannot require grad) and its weight
    # would differ from the reference.
    @pytest.mark.parametrize("reentrant", [True, False], ids=["reentrant", "non_reentrant"])
    @pytest.mark.parametrize(
        "module_name, layers_per_stage, checkpoints_per_stage, parts, checkpoints_per_rank, embedding_first", [
            ("PipelineModule", [2, 4], [1, 3], [0, 4, 12], [1, 3], False),
            ("DualPipeVModule", [1, 2, 2, 1], [1, 0, 2, 0], [0, 2, 6, 10, 12], [1, 2], False),
            ("PipelineModule", [2, 4], [1, 3], [0, 5, 13], [1, 3], True),
        ])
    def test_matches_sequential_model(self, module_name, layers_per_stage, checkpoints_per_stage, parts,
                                      checkpoints_per_rank, embedding_first, reentrant):
        """The stages hold the requested layers, only the requested blocks are checkpointed, and training gives the
        same loss and weights as the unpartitioned model, with reentrant and with non-reentrant checkpointing.
        """
        from deepspeed.pipe import DualPipeVModule
        module_cls = {"PipelineModule": PipelineModule, "DualPipeVModule": DualPipeVModule}[module_name]

        num_ranks = dist.get_world_size()
        micro_batches = 2 * num_ranks + 1
        hidden, vocab, lr = 8, 16, 0.1
        config = {
            "train_batch_size": micro_batches,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": micro_batches,
            "optimizer": {
                "type": "SGD",
                "params": {
                    "lr": lr
                }
            },
            "zero_allow_untested_optimizer": True,
        }
        torch.manual_seed(0)
        layers = [nn.Embedding(vocab, hidden)] if embedding_first else []
        for _ in range(6):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        reference = copy.deepcopy(nn.Sequential(*layers))
        data = [(torch.randint(0, vocab, (1, )) if embedding_first else torch.randn(1, hidden), torch.randn(1, hidden))
                for _ in range(micro_batches)]
        loss_fn = nn.MSELoss()

        checkpointed_blocks = []

        checkpoint = ds_checkpointing.checkpoint if reentrant else ds_checkpointing.non_reentrant_checkpoint

        def counting_checkpoint(function, *args):
            checkpointed_blocks.append(function)
            return checkpoint(function, *args)

        model = module_cls(layers=layers,
                           num_stages=num_ranks,
                           loss_fn=loss_fn,
                           partition_method='type:linear',
                           activation_checkpoint_interval=1,
                           activation_checkpoint_func=counting_checkpoint,
                           layers_per_stage=layers_per_stage,
                           activation_checkpoints_per_stage=checkpoints_per_stage)
        assert model.partitions() == parts
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        engine.set_dataiterator(RepeatingLoader(data))

        def reference_step():
            losses = []
            for x, y in data:
                loss = loss_fn(reference(x), y)
                (loss / micro_batches).backward()
                losses.append(loss.detach())
            with torch.no_grad():
                for p in reference.parameters():
                    p -= lr * p.grad
                    p.grad = None
            return torch.stack(losses).mean()

        reference_params = dict(reference.named_parameters())
        for _ in range(2):
            loss = engine.train_batch()
            assert torch.allclose(loss.cpu(), reference_step(), atol=1e-5)
        # the weights are compared after both steps: a rank that failed after the first step would leave the other rank
        # waiting in the second one
        for name, p in engine.module.named_parameters():
            assert torch.allclose(p.detach().cpu(), reference_params[name], atol=1e-5), name

        # every micro-batch of the two steps ran the checkpointed blocks of this rank's stages once
        assert len(checkpointed_blocks) == 2 * micro_batches * checkpoints_per_rank[dist.get_rank()]

    @pytest.mark.parametrize(
        "partition_method, layers_per_stage, checkpoints_per_stage, error",
        [
            ('type:linear', [2, 4], [1, -1], "non-negative integers"),
            ('type:linear', [2, 4], [5, 9], "checkpointable blocks"),
            ('type:linear', [2, 4], [1, 9], "checkpointable blocks"),
            # every layer counts, but a Tanh has no parameter and is not checkpointable: stage 0 holds two Linear and
            # two Tanh layers, and only its own ranks have built them
            ('uniform', [4, 8], [3, 4], "checkpointable blocks"),
        ])
    def test_rejects_invalid_checkpoint_counts(self, partition_method, layers_per_stage, checkpoints_per_stage, error):
        """Negative counts fail when the module is built; more recomputed blocks than a stage has fail when the
        pipeline engine is built, before any pipeline traffic and on every rank, also when only one stage's count is
        too large: a rank that went on alone would wait for the stage that raised."""
        hidden = 8
        layers = []
        for _ in range(6):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        config = {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 2,
            "optimizer": {
                "type": "SGD",
                "params": {
                    "lr": 0.1
                }
            },
            "zero_allow_untested_optimizer": True,
        }
        with pytest.raises(ValueError, match=error):
            model = PipelineModule(layers=layers,
                                   num_stages=dist.get_world_size(),
                                   loss_fn=nn.MSELoss(),
                                   partition_method=partition_method,
                                   activation_checkpoint_interval=1,
                                   layers_per_stage=layers_per_stage,
                                   activation_checkpoints_per_stage=checkpoints_per_stage)
            engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
            engine.set_dataiterator(RepeatingLoader([(torch.randn(1, hidden), torch.randn(1, hidden))] * 2))
            engine.train_batch()

    @staticmethod
    def _engine_config(pipeline=None):
        config = {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 2,
            "optimizer": {
                "type": "SGD",
                "params": {
                    "lr": 0.1
                }
            },
            "zero_allow_untested_optimizer": True,
        }
        if pipeline is not None:
            config["pipeline"] = pipeline
        return config

    @pytest.mark.parametrize("pipeline_config, raises", [
        (None, True),
        ({
            "activation_checkpoint_interval": 1
        }, False),
    ])
    def test_rejects_counts_without_checkpoint_interval(self, pipeline_config, raises):
        """With no checkpoint interval, neither the module's nor the config's, nothing is checkpointed and counts
        above 0 would be ignored without a word: the engine rejects them. An interval from the config is enough."""
        layers = [nn.Linear(8, 8) for _ in range(4)]
        model = PipelineModule(layers=layers,
                               num_stages=dist.get_world_size(),
                               loss_fn=nn.MSELoss(),
                               partition_method='type:linear',
                               activation_checkpoint_interval=0,
                               activation_checkpoints_per_stage=[1, 1])
        config = self._engine_config(pipeline_config)
        if raises:
            with pytest.raises(ValueError, match="activation checkpoint interval above 0"):
                deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        else:
            engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
            assert engine.module.activation_checkpoint_interval == 1

    @pytest.mark.parametrize("partition_method, checkpointable_layers, use_reentrant, warns", [
        ("uniform", None, True, True),
        ("type:linear", None, True, False),
        ("uniform", ["Linear"], True, False),
        ("uniform", None, False, False),
    ])
    def test_warns_when_counts_may_include_an_embedding(self, monkeypatch, partition_method, checkpointable_layers,
                                                        use_reentrant, warns):
        """Without a "type:" partition or checkpointable_layers, the counts include every block with parameters, an
        embedding too, which reentrant checkpointing leaves without gradients: rank 0 says so once the engine has set
        the checkpoint function (non-reentrant checkpointing from the config does not need the warning)."""
        import deepspeed.runtime.pipe.module as pipe_module
        warnings = []
        monkeypatch.setattr(pipe_module.logger, "warning", lambda message, *args: warnings.append(message))
        layers = [nn.Embedding(16, 8)] + [nn.Linear(8, 8) for _ in range(3)]
        model = PipelineModule(layers=layers,
                               num_stages=dist.get_world_size(),
                               loss_fn=nn.MSELoss(),
                               partition_method=partition_method,
                               activation_checkpoint_interval=1,
                               checkpointable_layers=checkpointable_layers,
                               activation_checkpoints_per_stage=[1, 1])
        deepspeed.initialize(config=self._engine_config({"use_reentrant": use_reentrant}),
                             model=model,
                             model_parameters=model.parameters())
        said = any("activation_checkpoints_per_stage counts every block" in w for w in warnings)
        assert said == (warns and dist.get_rank() == 0)

    def test_checkpoint_loads_into_another_split(self, tmpdir):
        """Layer files are named by the global layer index, so the weights saved with one split load into another."""
        hidden = 8
        config = {
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 4,
            "optimizer": {
                "type": "SGD",
                "params": {
                    "lr": 0.1
                }
            },
            "zero_allow_untested_optimizer": True,
        }

        def linear_tanh_layers(seed):
            torch.manual_seed(seed)
            layers = []
            for _ in range(6):
                layers += [nn.Linear(hidden, hidden), nn.Tanh()]
            return layers

        def engine_for(layers, **kwargs):
            model = PipelineModule(layers=layers,
                                   num_stages=dist.get_world_size(),
                                   loss_fn=nn.MSELoss(),
                                   partition_method='type:linear',
                                   activation_checkpoint_interval=1,
                                   **kwargs)
            engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
            return engine

        generator = torch.Generator().manual_seed(1)
        data = [(torch.randn(1, hidden, generator=generator), torch.randn(1, hidden, generator=generator))
                for _ in range(4)]
        saved = engine_for(linear_tanh_layers(seed=0))
        saved.train_batch(iter(data))
        expected = saved.eval_batch(iter(data)).item()
        saved.save_checkpoint(tmpdir)

        # other weights, so only the checkpoint can make the losses agree
        loaded = engine_for(linear_tanh_layers(seed=7),
                            layers_per_stage=[2, 4],
                            activation_checkpoints_per_stage=[1, 3])
        assert loaded.module.partitions() != saved.module.partitions()
        assert loaded.eval_batch(iter(data)).item() != pytest.approx(expected, rel=1e-3)
        loaded.load_checkpoint(tmpdir, load_optimizer_states=False, load_lr_scheduler_states=False)
        assert loaded.eval_batch(iter(data)).item() == pytest.approx(expected, rel=1e-5)
