Deep500 For MAELSTROM
====================================================================

## Introduction

### Deep500

![Deep500](deep500.svg)
<br />
(or: 500 ways to train deep neural networks)


Deep500 is a library that can be used to customize and measure anything with deep neural networks, using a clean, high-performant, and simple interface. 

Using Deep500, you automatically gain:
* Operator validation, including gradient checking for backpropagation
* Statistically-accurate performance benchmarks and plots
* High-performance integration with popular deep learning frameworks (see Supported Frameworks below)
* Running your operator/framework/optimizer/communicator/... with real workloads, alongside existing environments
* and much more...

### MAELSTROM D2.6

Delivarable 2.6 aims is to perform software benchmarking of the applications inside MAELSTROM. To this end, we provide an easy-to-use API on top of Deep500 to measure different aspects of the applications __with minimum efforts__. In particular, it does _not_ require you to convert your application to use Deep500 recipes or operators.
Instead, you can modify it in-place.


Note about using Deep500 in Julich JURECA-DC - When trying to install deep500 on Jureca using a similar software stack as we had used in previous benchmarking runs, I noticed that the latest Tensorflow version available in the server (TensorFlow/2.11.0-CUDA-11.7), led to a conflict with the protobuf version. That version of TF requires the use of some other modules including protobuf 3.19.4, while deep500 was trying to install newer versions like 4.25.2. This is coming from the onxx version that is installed with deep500, which tries to install the latex onnx (1.15.) requiring an updated protobuf version. This problem can be solved if ones install onnx 1.11, allowing developers to use the TF version available in JURECA. 

Note about using Deep500 with MAELSTROM AP3 - For AP3, we encountered an issue when testing Deep500, due to the fact that the tensorflow dataset doesn't have a predefined length. This can be  solved by providing the ‘steps_per_epoch’ explicitly 


## Installation

Make sure that you do not have any other version of Deep500 installed. If you do, please uninstall it first.

git clone https://github.com/sashkboos/Deep500-for-MAELSTROM.git deep500
cd deep500
pip install -e .


## Usage

### Importing

First, you should import the appropriate type of timers for the framework you are using.

PyTorch:
```py
from deep500.utils import timer_torch as timer
```

TensorFlow:
```py
from deep500.utils import timer_tf as timer
```

Any other framework:
```py
from deep500.utils import timer
```

----

### Timing Code

**Note:** If you are using TensorFlow's `model.fit`, [see below](#tensorflow-callback) on a simplified callback for this.

A timer is created as follows:
```py
tmr = timer.Timer()
```
The timers are stateful objects, as they maintain a log of what you have timed, so in general you should pass them via arguments (or as class members) rather than creating new ones.

To time something, you simply define the region you wish to measure using a simple start and stop paradigm.
With each region, you associate a predefined _key_, which semantically lables what you are timing.
See below for a list of keys.
(Note: You may nest or overlap regions which use different keys, but not which use the same key. You should get an exception if you attempt this.)
A region is opened by calling `tmr.start(key)` and closed by calling `tmr.end(key)`.

For example, to introduce a timer around each epoch:
```py
for epoch in range(num_epochs):
  tmr.start(timer.TimeType.EPOCH)
  train_epoch(loader, net, criterion, optimizer, tmr)
  tmr.end(timer.TimeType.EPOCH)
```

When this code is run, the time spent in the region will be automatically measured and recorded.
All times are reported in fractional seconds.

The available keys for timing regions are:
* `timer.TimeType.EPOCH` - one complete pass over a dataset
* `timer.TimeType.BATCH` - one mini-batch
* `timer.TimeType.FORWARD` - forward propagation for one mini-batch
* `timer.TimeType.BACKWARD` - backward propagation for one mini-batch
* `timer.TimeType.COMM` - communication during one mini-batch
* `timer.TimeType.IO` - I/O to load data for one mini-batch
* `timer.TimeType.OTHER` - a catch-all for a user-defined region

### Saving Timing Results

There are several ways to retrieve or save timing results.
* `tmr.get_time(key)` will return a list of all times recorded for `key`.
* `tmr.get_time_stats(key)` will return an object with various summary statistics for all times recorded for `key`.
* `tmr.print_all_time_stats()` will print (to `stdout`) summary statistics for all times recorded for all keys.
* `tmr.save_all_time_stats(filename)` is similar, but will write the output to the file named by `filename` (which must not exist).
* `tmr.log_wb_all(prefix='')` will log the mean time of all keys recorded to Weights & Biases (optionally prepending `prefix` to the logging keys).
* `tmr.log_mlflow_all(prefix='')` is similar, but will log to MLFlow instead.

To collect overall benchmarking results, we recommend using `tmr.print_all_time_stats()` (or `tmr.save_all_time_stats()`) to get maximum details.
Logging to W&B or MLFlow is a good way to additionally monitor performance.

### Timing GPU Code with PyTorch or TensorFlow

Code running on the GPU typically executes asynchronously (i.e., the CPU launches a series of kernels, but does not wait for their computations to complete, only checking later).
This means that timing solely on the CPU may not adequately reflect the actual computation time for certain regions.
When using PyTorch or TensorFlow, we also support adding timers for GPU kernels.
For PyTorch, this is low-overhead; due to technical issues, there may be additional overheads with TensorFlow.

When using the these timers (`from deep500.utils import timer_{torch,tf} as timer`), in addition to the standard `Timer` class, you can use the `CPUGPUTimer` class.
This class has exactly the same API as the standard `Timer`, except methods for timing regions take an additional, optional `gpu` keyword argument.
This is a boolean, defaulting to `False`, which controls whether to time on the GPU.
For `start` and `end`, if this is `False`, timing is only performed on the CPU; if it is `True`, timing is performed on both the CPU and GPU.
Note that you should start and end a region using the same `gpu` argument in both.
For `get_time` and `get_time_stats`, this controls whether to get timing results for the CPU or GPU for the provided key; the other methods which return all times or statistics always operate on both CPU and GPU times, which are reported separately.

In the above example, we recommend to add GPU timing to the `BATCH`, `FORWARD`, and `BACKWARD` regions.

----

#### Technical note

GPU timing is implemented using CUDA events.
Calling `end` simply marks the end of a timing region and does not actually measure the time, as that would require synchronizing the CPU and GPU.
Thus, if you time many regions, you may use many CUDA events, which are a finite resource.
This can happen, for example, when there are a large number of batches in an epoch, and you are timing each of them.

Whenever you request times (e.g., `get_time`, `print_all_time_stats`, etc.), outstanding timers are completed so the times can be recorded.
Alternately, you can manually complete timers by calling `tmr.complete_all`.
It is recommended to call this occasionally (e.g., every 20 batches) to ensure you do not exhaust the available CUDA events.
For example:
```py
if batch_idx % 20 == 0:
    tmr.complete_all()
```

----

#### Technical note for TensorFlow

When using the TensorFlow GPU timer, we use CuPy to provide CUDA events.
However, because the stream used by TensorFlow to launch kernels is not exposed, we synchronize on the default stream, which may cause additional overhead.
Further, this synchronization cost is incurred every time `end` is called.

Additionally, TensorFlow's API may limit how fine-grained timing can be, as, e.g., backpropagation may not be broken out as nicely as it is in PyTorch.
In this case, it is probably best to limit timing to batch or epoch granularity.
----

### A More Complete Example

This is a more detailed example, showing how you can add timing to a prototypical PyTorch training loop:
```py
import torch
# Adjust import depending on your model.
from deep500.utils import timer_torch as timer


def train_epoch(loader, net, criterion, optimizer, tmr):
    # Batches, forward, and backward are timed on both CPU and GPU.
    # Because we want the batch time to include I/O, we start timing here.
    # Timing for each subsequent iteration is handled at the end of the loop.
    tmr.start(timer.TimeType.BATCH, gpu=True)  # Timing region for one batch.
    # Timing I/O operates similar to batch timing.
    # Note: This only times the visible delay for I/O, not overall I/O time.
    tmr.start(timer.TimeType.IO)  # Timing region for I/O for this batch.
    for idx, (inputs, targets) in enumerate(loader):
        tmr.end(timer.TimeType.IO)
        tmr.start(timer.TimeType.FORWARD, gpu=True)  # Timing region for forward prop.
        output = net(inputs)
        loss = criterion(output, targets)
        tmr.end(timer.TimeType.FORWARD, gpu=True)
        tmr.start(timer.TimeType.BACKWARD, gpu=True)  # Timing region for backprop.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tmr.end(timer.TimeType.BACKWARD, gpu=True)

        # See above on batch and I/O timing.
        # We start timing the next batch except at the last iteration.
        tmr.end(timer.TimeType.BATCH, gpu=True)
        if idx % 10 == 0:
            # Prevent CUDA event exhaustion.
            tmr.complete_all()
        if idx != len(loader) - 1:
            tmr.start(timer.TimeType.BATCH, gpu=True)
            tmr.start(timer.TimeType.IO)

def train_model(args, loader, net, criterion, optimizer):
    net.train()
    tmr = timer.CPUGPUTimer()  # Create timer.
    # Note: Use timer.Timer() for CPU-only timing.
    for epoch in range(args.num_epochs):
        tmr.start(timer.TimeType.EPOCH)  # Timing region for one epoch.
        # Note we pass the timer as a parameter, rather than creating a new one.
        train_epoch(loader, net, criterion, optimizer, tmr)
        tmr.end(timer.TimeType.EPOCH)
    tmr.print_all_time_stats()  # Print all gathered stats to stdout.
    tmr.log_mlflow_all('train')  # Log the means of each region type to MLFlow.
```

### TensorFlow Callback

When using the TensorFlow `model.fit()` approach to training, there may be no exposed training loop to time.
For this case, we provide a callback that supports epoch- and batch-level timing during training.

The basic usage is as follows:
```py
from deep500.utils import timer_tf as timer

tmr = timer.CPUGPUTimer()
model.fit(
    # Other args...
    callbacks=[timer.TimerCallback(tmr, gpu=True)]
)
```

The `TimerCallback` takes two arguments in its constructor:
1. An instance of `CPUGPUTimer`. (This is passed in so you can later save the timing results from the instance.)
2. Whether to time batches on GPU (set `gpu=True`).

## API Reference

Below is a brief API reference for the `Timer` class.
For full details, refer to [the source](https://github.com/ndryden/deep500/tree/master/deep500/utils).

_Note_: All times reported are in units of fractional seconds.

#### `Timer.start(key: TimeType) -> None`
Start timing a region with type given by `key`.

#### `Timer.end(key: TimeType) -> None`
End timing a region with type given by `key`.

#### `Timer.get_time(key: TimeType) -> list[float]`
Return a list of all recorded times for regions of type `key`.

#### `Timer.get_time_stats(key: TimeType) -> TimeStats`
Return summary statistics for all recorded times for regions of type `key`.

The statistics are in a `TimeStats` class, which has the following members:
* `min`
* `mean`
* `median`
* `max`
* `stdev`

(If fewer than two times are recorded, the standard deviation is treated as 0.)

#### `Timer.print_all_time_stats(file: Any = None) -> None`
Print summary statistics for all region types where time has been recorded.

Optionally, these can be printed to `file` (which must have a `write` method -- see Python's documentation for `print`.)

#### `Timer.save_all_time_stats(filename: str) -> None`
Save summary statistics for all region types where time has been recorded to a file with name `filename`.

Note `filename` must not exist.

#### `Timer.log_wb_all(prefix: str = '') -> None`
Log the mean time for each region type where time has been recorded to Weights & Biases.

`prefix` is an optional string to prepend to the logged keys (e.g., `train/`).

This requires W&B to be installed and initialized.

#### `Timer.log_mlflow_all(prefix: str = '') -> None`
Log the mean time for each region type where time has been recorded to MLFlow.

`prefix` is an optional string to prepend to the logged keys (e.g., `train/`).

This requires MLFlow to be installed and initialized.

#### `Timer.complete_all() -> None`
Complete all outstanding timing. (See GPU timing above.)


