---
title: "Pipeline Parallelism"
tags: training large-model
---

DeepSpeed v0.3 includes new support for pipeline parallelism! Pipeline
parallelism improves both the memory and compute efficiency of deep learning
training by partitioning the layers of a model into stages that can be
processed in parallel.
DeepSpeed's training engine provides hybrid data and pipeline parallelism and
can be further combined with model parallelism such as
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
An illustration of
3D parallelism is shown below. Our latest [results]({{ site.press_release_v3 }})
demonstrate that this 3D parallelism enables training models with over a
**trillion** parameters.

![3D parallelism in DeepSpeed](/assets/images/3d-parallelism.png)

DeepSpeed uses *gradient accumulation* to extract pipeline parallelism (shown
below). Each batch of training data is divided into micro-batches that can be
processed in parallel by the pipeline stages. Once a stage completes the
forward pass for a micro-batch, the activation memory is communicated to the
next stage in the pipeline. Similarly, as the next stage completes its
backward pass on a micro-batch, the gradient with respect to the activation
is communicated backwards through the pipeline. Each backward pass
accumulates gradients locally. Next, all data parallel groups perform
reductions of the gradients in parallel. Lastly, the optimizer updates the
model weights.


Below is an illustration of how DeepSpeed will train a batch with eight
micro-batches using hybrid two-way data parallelism and two-stage pipeline
parallelism. GPUs 0 and 2 are arranged in a pipeline and will alternate
forward (F) and backward (B) passes. They will then all-reduce (AR) gradients
with their data parallel counterparts, GPUs 1 and 3, respectively. Finally,
the two pipeline stages update their model weights.

![Pipeline Schedule](/assets/images/pipe-schedule.png)


## Getting Starting with Pipeline Parallelism

DeepSpeed strives to accelerate *and* simplify the process of pipeline
parallel training. This section provides first steps with hybrid data and
pipeline parallel training by preparing `torchvision`'s
[AlexNet](https://pytorch.org/docs/1.2.0/_modules/torchvision/models/alexnet.html)
model.

### Expressing Pipeline Models
Pipeline parallelism requires models to be expressed as a sequence of layers.
In the forward pass, each layer consumes the output of the previous
layer. In fact, there is no need to specify a `forward()` for a pipeline
parallel model! The forward pass of a pipeline parallel model implicitly
takes the form:
```python
def forward(self, inputs):
    x = inputs
    for layer in self.layers:
        x = layer(x)
    return x
```

PyTorch's
[`torch.nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
is a convenient container for expressing pipeline parallel models and can be
parallelized by DeepSpeed with no modification:
```python
net = nn.Sequential(
    nn.Linear(in_features, hidden_dim),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_dim, out_features)
)
from deepspeed.pipe import PipelineModule
net = PipelineModule(layers=net, num_stages=2)
```
`PipelineModule` uses its `layers` argument as the sequence of layers that
comprise the model. After initialization, `net` is divided into two pipeline
stages and its layers moved to the corresponding GPUs. If more than two GPUs
are present, DeepSpeed will also use hybrid data parallelism.

**Note:** The total number of GPUs must be divisible by the number of pipeline
stages.
{: .notice--info}

**Note:** For large model training, see [memory-efficient model construction](#memory-efficient-model-construction).
{: .notice--info}

### AlexNet
Let's look at an abbreviated implementation of `torchvision`'s
[AlexNet](https://pytorch.org/docs/1.2.0/_modules/torchvision/models/alexnet.html):

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            ...
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            ...
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```
`AlexNet` is mostly a composition of several `Sequential` submodules. We can
turn this into a `PipelineModule` by flattening its submodules into a single
sequence of layers:
```python
class AlexNetPipe(AlexNet):
    def to_layers(self):
        layers = [
            *self.features,
            self.avgpool,
            lambda x: torch.flatten(x, 1),
            *self.classifier
        ]
        return layers

from deepspeed.pipe import PipelineModule
net = AlexNetPipe()
net = PipelineModule(layers=net.to_layers(), num_stages=2)
```

**Note:**
the `lambda` in the middle of `layers` above is not a `torch.nn.Module`
type. Any object that implements `__call__()` can be a layer in a
`PipelineModule`: this allows for convenient data transformations in the
pipeline.
{: .notice--info}


### Inputs and Outputs
Following `torch.nn.Sequential`, the inputs and outputs of each layer must be
either a single `torch.Tensor` or a `tuple` of tensors. In practice, some
models may need to modify their forward pass to pack and unpack arguments to
`forward()`. Consider an abbreviated implementation of a stack of Transformer
blocks:
```python
class TransformerBlock(nn.Module)
    ...
    def forward(self, hidden, mask):
        output = self.compute(hidden, mask)
        return output
    ...

stack = [ TransformerBlock() for _ in range(num_layers) ]
```
Two modifications to `TransformerBlock` are required:

1. The arguments must be collected into a `tuple`.
2. `mask` must also be returned from `forward()` to pass to the next layer.

These modifications can be accomplished with a short subclass:
```python
class TransformerBlockPipe(TransformerBlock)
    def forward(self, inputs):
        hidden, mask = inputs
        output = super().forward(hidden, mask)
        return (output, mask)
stack = [ TransformerBlockPipe() for _ in range(num_layers) ]
```

### Training Loops

Pipeline parallelism interleaves forward and backward passes, and thus the
training loop cannot be divided into separate stages of `forward()`,
`backward()` and `step()`.
Instead, DeepSpeed's pipeline engine provides a `train_batch()` method that
advances the pipeline engine until the next batch of training data is
consumed and the model weights updated.
```python
train_iter = iter(train_loader)
loss = engine.train_batch(data_iter=train_iter)
```

The above `train_batch()` example is equivalent to the following with
traditional data parallel DeepSpeed:
```python
train_iter = iter(train_loader)
for micro_batch in engine.gradient_accumulation_steps():
    batch = next(data_iter)
    loss = engine(batch)
    engine.backward(loss)
    engine.step()
```

### Dealing with Data

Data parallel training typically has each worker perform IO independently at
the start of each batch. However, in a pipeline parallel environment, only the
first stage uses the input data, and only the last stage uses labels for loss
calculation.

**Note:**
The pipeline engine expects data loaders to return a `tuple` of two items. The
first returned item is the input batch data, and the second item is the data
to be used in the loss calculation. As before, inputs and labels should be
either `torch.Tensor` type or a `tuple` of tensors.
{: .notice--info}

For convenience, the DeepSpeed pipeline engine can construct a distributed
data loader when a dataset is provided to `deepspeed.initialize()`. DeepSpeed
handles the rest of the complexity of data loading, and so the pipeline
training loop becomes:
```python
engine, _, _, _ = deepspeed.initialize(
    args=args,
    model=net,
    model_parameters=[p for p in net.parameters() if p.requires_grad],
    training_data=cifar_trainset())

for step in range(args.steps):
    loss = engine.train_batch()
```

Of course, DeepSpeed will work with any data loader that you wish to use.
Data loaders should be constructed by the first and last stages in the
pipeline. Each worker should load micro-batches of size
`engine.train_micro_batch_size_per_gpu()` and will be queried
a total of `engine.gradient_accumulation_steps()` times per `train_batch()`.

**Watch out!**
The pipeline engine *pulls* data from an iterator instead of iterating over
it. It's critical that the data stream does not empty in the middle of a
training batch. Each invocation of `train_batch()` will pull
a total of `engine.gradient_accumulation_steps()` micro-batches of data from
the data iterator.
{: .notice--warning}

DeepSpeed provides a convenience class `deepspeed.utils.RepeatingLoader` that
simply wraps an iterable such as a data loader and restarts it whenever the
end is reached:
```python
train_loader = deepspeed.utils.RepeatingLoader(train_loader)
train_iter = iter(train_loader)
for step in range(args.steps):
    loss = engine.train_batch(data_iter=train_iter)
```


## Advanced Topics


### Load Balancing Pipeline Modules
The performance of pipeline parallel training strongly relies on load
balance. DeepSpeed provides several mechanisms for partitioning the model
across GPUs. These strategies can be set with the `partition_method` keyword
argument to `PipelineModule`. Here are partitioning methods currently provided
by DeepSpeed:

* `partition_method="parameters"` (**default**)
   balances the number of trainable parameters on each pipeline stage . This is
   especially useful in memory-constrained environments and when the size of a
   layer is proportional to the computation time.
* `partition_method="type:[regex]"`
  balances layers whose class names match `[regex]`. The regular expression
  is not case sensitive. For example, `partition_method="type:transformer"`
  would balance the number of transformer layers per stage.
* `partition_method="uniform"` balances the number of layers per stage.

A balanced split is not always the fastest one: the first stage holds the
activations of more micro-batches than the last stage under the 1F1B
schedule, and the first and last stages also hold the embedding and the
output layers. The `layers_per_stage` keyword argument sets the number of
layers of each stage instead of balancing them. It counts every layer with
`partition_method="uniform"`, and only the layers whose class names match with
`partition_method="type:[regex]"`:

```python
# 6, 8, 8 and 10 transformer layers on the four stages
net = PipelineModule(layers=layers,
                     num_stages=4,
                     partition_method="type:transformer",
                     layers_per_stage=[6, 8, 8, 10])
```

In the same way, `activation_checkpoints_per_stage` sets how many of the
checkpointable blocks of `activation_checkpoint_interval` layers each stage
checkpoints, counted from the first block of the stage; the other blocks of the
stage keep their activations. For example, with
`activation_checkpoint_interval=1` and
`activation_checkpoints_per_stage=[6, 4, 0, 0]`, the first stage recomputes all
6 of its transformer layers, the second stage 4 of its 8, and the last two
stages none.

A pipeline checkpoint stores one file per layer, named by the layer's global
index, so the weights saved with one split load into a model with another
`layers_per_stage`: pass `load_optimizer_states=False` (or
`load_module_only=True`) to `load_checkpoint`. The optimizer state is saved per
rank for the layers that rank held, and loading it into a different split
fails with an error.

#### Choosing the split with the planner

`deepspeed.runtime.pipe.planner` chooses `layers_per_stage` and
`activation_checkpoints_per_stage` from measurements of the model's layer
types, so that every stage fits its device and the slowest stage is as fast
as possible. Its search is the one of X-MoE-4D's pipeline planner (ELM-PP).
It has three parts:

1. **Profile.** `profile_layer_types` builds one layer of each type, outside
   the training loop, and measures its forward and backward time with and
   without recompute, the memory a micro-batch keeps until its backward, the
   peaks of the forward and of the backward, its parameter list and the size
   of its buffers. `write_costs` stores the result in a costs file.
2. **Predict.** `StageModel` computes, for any split, what every stage
   allocates at the three peaks of a step (the 1F1B schedule, the gradient
   reduction and the optimizer step) and how long the step takes. The memory
   terms are either computed from what DeepSpeed allocates (ZeRO stage 1,
   bf16, the 1F1B schedule) or taken from the costs file; none is fitted to a
   run, and each one is described in `planner/memory.py`. Where the code
   allows a range the model takes the upper end, so a stage is predicted to
   need at least what it uses.
3. **Search.** A stage gets the fewest recomputed blocks with which it fits
   the budget. `plan` returns the split whose slowest stage is fastest, found
   by a dynamic program over the number of blocks on the first stages. Under
   1F1B the first micro-batch crosses every stage once and each further
   micro-batch follows after the time of the slowest stage, so with many
   micro-batches per step the slowest stage sets the step time. Among several
   micro-batch sizes the planner takes the one with the shortest step.

A stage fits when its `reserve` is within the budget, and `reserve` is more
than the largest allocation. PyTorch's allocator keeps a freed block and cuts
it for later, smaller requests, and it cannot move a live tensor. The gradient
reduction and the optimizer step allocate a few large tensors that the 1F1B
phase does not hold; if the 1F1B phase grows into the space they left, the
next reduction can find no block of the size it needs although as much memory
is free. So `reserve` lays the three phases side by side.

```bash
python -m deepspeed.runtime.pipe.planner plan --costs costs.json \
    --layers 'Embed,Block*32,Head' --blocks Block \
    --world 32 --pp 4 --ep 8 --global-batch 128 --device-gib 64
```

The command prints, for every micro-batch size in `--mbs-list` (by default
every size that the costs file holds), the best split, the time and the memory of each stage, and a line `BEST {...}`
with the `layers_per_stage` and `activation_checkpoints_per_stage` to pass to
`PipelineModule`. After it come two reference lines for the even split at the
same micro-batch size, with no block and with every block recomputed.
`predict` prints the same for a split given with `--parts` and `--ckpts`
(the even split by default). `balance-memory` returns the split whose largest
stage memory is smallest when every block is recomputed (`--recompute all`)
or none, for the largest micro-batch size that fits.

The step time adds two terms to the measured layer times, as in X-MoE-4D:
`--overhead-ms`, the time every stage spends on a micro-batch outside its
layers, and `--optimizer-ratio`, the optimizer step as a share of the forward
and backward passes. Their defaults, 55 ms and 0.10, are the defaults of
X-MoE-4D's planner (`mb_framework_overhead_ms` and `optimizer_step_ratio` in
its `ELM_planner.py`), where the 55 ms stands for kernel launches, the MoE
all-to-all latency and the pipeline's send and receive of every micro-batch.
Neither changes the split. The predicted step is a guide, not a measurement:
on a 63B model with DeepSpeed's MoE layer (fixed capacity) it was 21.0 s
against 16.8 s measured, because the layer times measured here already contain
the all-to-all (18.1 s with both terms set to 0); with AutoEP's MoE layer and
an untrained router, whose busiest expert-parallel rank paces every MoE layer,
it was 25.9 s against about 29.8 s. The overhead matters when several
micro-batch sizes are compared, because a larger size has fewer micro-batches
per step, so set it from a run of your own model when you pass more than one
size. The optimizer share scales every step time alike.

`--headroom-gib` (default 4) is the device memory the plan may
not use: what the device holds outside PyTorch's allocator (its context and the
communication library; the used memory that `nvidia-smi` or `rocm-smi` reports
minus `torch.cuda.memory_reserved()`, 3.0 GiB on the AMD MI250X GPUs of the
example's results), plus a margin. X-MoE-4D keeps 2.5 GiB free, but its memory
formula predicts 7 to 25 GB more than a stage uses, while this model
predicted 0.4 to 1.6 GiB more on the 63B runs, so here the headroom must hold
what lies outside the allocator. `--inplace-optimizer` tells the model that
the optimizer updates in place, as DeepSpeed's FusedAdam does; without it the
model leaves room for the temporaries of `torch.optim`'s Adam and AdamW.
`--input-bytes` and `--label-bytes` add what a micro-batch's inputs and labels
hold on the device; token ids are small enough to leave out.

The model covers ZeRO stage 1 with bf16 parameters, gradients and
communication, `activation_checkpoint_interval=1`, a
`partition_method="type:[regex]"` whose type is the block that `--blocks`
names, and layers that take one tensor and return one tensor. It has been
validated without tensor parallelism. With an MoE layer that drops no token, such as
AutoEP's, a block's memory follows the rows its rank receives, and in training
the busiest expert-parallel rank receives more than the mean. Profile such a
block's memory with its rank receiving the imbalance you expect: on a 63B model
whose busiest rank received 2.2 to 2.4 times the mean rows, a profile at the
router's own routing on random inputs predicted 2.6 to 6.0 GiB per stage too
little, and one at 2.5 times the mean predicted 1.1 to 2.8 GiB more than the
stages allocated. The DeepSpeedExamples profiler does this with
`--routing-imbalance`. A complete example,
with a profiling script and a training script that prints the measured memory
of each stage next to the prediction, is in
[DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/training/pipeline_planner).

### Memory-Efficient Model Construction
Building a `Sequential` container and providing it to a `PipelineModule` is a convenient way
of specifying a pipeline parallel model. However, this approach encounters scalability issues
for massive models because each worker replicates the whole model in CPU memory.
For example, a machine with 16 GPUs must have as much local CPU memory as 16 times the model size.

DeepSpeed provides a `LayerSpec` class that delays the construction of
modules until the model layers have been partitioned across workers.
Then each worker will allocate only the layers it's assigned to. So, comparing to the
example from the previous paragraph, using `LayerSpec` a machine with 16 GPUs will need to
allocate a total of 1x model size on its CPU memory and not 16x.

Here is an example of the abbreviated AlexNet model, but expressed only
with `LayerSpec`s. Note that the syntax is almost unchanged: `nn.ReLU(inplace=True)`
simply becomes `LayerSpec(nn.ReLU, inplace=True)`.
```python
from deepspeed.pipe import PipelineModule, LayerSpec
class AlexNetPipe(PipelineModule):
    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        specs = [
            LayerSpec(nn.Conv2d, 3, 64, kernel_size=11, stride=4, padding=2),
            LayerSpec(nn.ReLU, inplace=True),
            ...
            LayerSpec(nn.ReLU, inplace=True),
            LayerSpec(nn.Linear, 4096, self.num_classes),
        ]
        super().__init__(layers=specs, loss_fn=nn.CrossEntropyLoss(), **kwargs)
```

### Tied Layers
Some models cannot be entirely expressed as pipeline parallel models because
some layers are reused in the pipeline. For example, Transformer based
language models commonly use an embedding layer early in the pipeline to map
vocabulary to hidden states, and then use the embedding to map hidden states
back to vocabulary at the end of the pipeline. If the model was restricted to
pure pipeline parallelism, this embedding reuse would prohibit pipeline
parallelism.

DeepSpeed provides a `TiedLayerSpec` that is an extension of
`LayerSpec`. `TiedLayerSpec` requires an additional argument: `key`.
Each reuse of a layer is specified with a `TiedLayerSpec`, and the `key` field
is used to identify where a layer is reused.

Tied layers are replicated on every pipeline stage that owns an instance of
reuse. Training then proceeds as normal, but an additional all-reduce of the
tied gradients is added after all backward passes complete. The all-reduce
ensures that the weights of the tied layer remain in sync across pipeline stages.

### DualPipeV Schedule
DualPipeV is a pipeline schedule from
[DualPipe](https://github.com/deepseek-ai/DualPipe) in which each GPU holds two
pipeline stages. With `P` GPUs in the pipeline the model is partitioned into
`2P` stages, and GPU `r` holds stage `r` and stage `2P - 1 - r`. A micro-batch
runs forward from GPU `0` to GPU `P - 1`, turns around there, and runs back to
GPU `0`. GPU `0` therefore loads the data and also computes the loss. While one
of a GPU's stages waits for its neighbor, the other stage can process a
different micro-batch.

To use DualPipeV, build the model with `DualPipeVModule` in place of
`PipelineModule`. It accepts the same arguments, and `num_stages` is still the
number of GPUs in the pipeline.
```python
from deepspeed.pipe import DualPipeVModule
net = DualPipeVModule(layers=net, loss_fn=torch.nn.CrossEntropyLoss(), num_stages=2)
engine, _, _, _ = deepspeed.initialize(config=config, model=net, model_parameters=net.parameters())
loss = engine.train_batch(data_iter=train_iter)
```

`deepspeed.initialize()` selects the DualPipeV engine when it is given a
`DualPipeVModule`. `train_batch()`, `eval_batch()`, activation checkpointing,
and saving and loading checkpoints are used as with `PipelineModule`.

DualPipeV has the following requirements:

* `gradient_accumulation_steps` must be at least `2P`. The same minimum applies
  to `num_micro_batches` in `eval_batch()`.
* `dynamic_shape=True` is not supported.
* Boolean tensors cannot be passed between stages.
* The `pipe_partitioned` and `grad_partitioned` options for model parallelism
  are not supported.
