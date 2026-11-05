# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Choose the layers and the recomputed blocks of each pipeline stage from measured layer costs.

The answer is the two ``PipelineModule`` arguments ``layers_per_stage`` and ``activation_checkpoints_per_stage``.

- ``profile``: measures each layer type once, outside the training loop, and writes a costs file;
- ``memory``: the memory of one stage at the three peaks of a step (ZeRO stage 1, bf16 parameters, Adam);
- ``search``: the search over splits and recompute counts, with the rules of X-MoE-4D's pipeline planner (ELM-PP).

The planner is X-MoE-4D's memory-aware pipeline planner. Its search rules and its step-time formulas are kept. Its
memory prediction is rebuilt from what DeepSpeed allocates, and its layer times are measured per layer type instead
of taken from timers inside a training run. Because a ``PipelineModule`` lists every layer, a stage always holds a
block and its time counts the layers that are not blocks (``search`` says where this departs from ELM-PP).
"""

from .memory import data_parallel_sizes, optimizer_groups, zero1_partition, StageModel
from .search import even_parts, plan, plan_memory_balanced, predict
from .profile import profile_layer_types, write_costs, load_costs
