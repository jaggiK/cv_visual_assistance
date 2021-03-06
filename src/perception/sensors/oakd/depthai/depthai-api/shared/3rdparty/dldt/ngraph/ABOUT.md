About nGraph Compiler stack
===========================

nGraph Compiler stack architecture
----------------------------------

The diagram below represents our current release stack. In the diagram, 
nGraph components are colored in gray. Please note
that the stack diagram is simplified to show how nGraph executes deep
learning workloads with two hardware backends; however, many other
deep learning frameworks and backends currently are functioning.

![](doc/sphinx/source/graphics/ngraph_arch_diag.png)


## Bridge

Starting from the top of the stack, nGraph receives a computational graph
from a deep learning framework such as TensorFlow or MXNet. The
computational graph is converted to an nGraph internal representation 
by a bridge created for the corresponding framework.

An nGraph bridge examines the whole graph to pattern match subgraphs
which nGraph knows how to execute, and these subgraphs are encapsulated.
Parts of the graph that are not encapsulated will default to framework
implementation when executed.


## nGraph Core

nGraph uses a strongly-typed and platform-neutral
`Intermediate Representation (IR)` to construct a "stateless"
computational graph. Each node, or op, in the graph corresponds to
one `step` in a computation, where each step produces zero or
more tensor outputs from zero or more tensor inputs.

This allows nGraph to apply its state of the art optimizations instead
of having to follow how a particular framework implements op execution,
memory management, data layouts, etc.

In addition, using nGraph IR allows faster optimization delivery
for many of the supported frameworks. For example, if nGraph optimizes
ResNet for TensorFlow, the same optimization can be readily applied
to MXNet* or ONNX* implementations of ResNet.


## Hybrid Transformer

Hybrid transformer takes the nGraph IR, and partitions it into
subgraphs, which can then be assigned to the best-performing backend.
There are two hardware backends shown in the stack diagram to demonstrate
this graph partitioning. The Hybrid transformer assigns complex operations
(subgraphs) to Intel?? Nervana??? Neural Network Processor (NNP) to expedite the
computation, and the remaining operations default to CPU. In the future,
we will further expand the capabilities of Hybrid transformer
by enabling more features, such as localized cost modeling and memory
sharing.

Once the subgraphs are assigned, the corresponding backend will
execute the IR.


## Backends

Focusing our attention on the CPU backend, when the IR is passed to
the Intel?? Architecture (IA) transformer, it can be executed in two modes:
Direct EXecution (DEX) and code generation (`codegen`).

In `codegen` mode, nGraph generates and compiles code which can
either call into highly optimized kernels like MKL-DNN or JITers like Halide.
Although our team wrote kernels for nGraph for some operations,
nGraph leverages existing kernel libraries such as MKL-DNN, Eigen, and MLSL.

MLSL library is called when nGraph executes distributed training.
At the time of the nGraph Beta release, nGraph achieved state of the art
results for ResNet50 with 16 nodes and 32 nodes for TensorFlow and MXNet.

The other mode of execution is Direct EXecution (DEX). In DEX mode,
nGraph can execute the operations by directly calling associated kernels
as it walks though the IR instead of compiling via `codegen`. This mode
reduces the compilation time, and it will be useful for training,
deploying, and retraining a deep learning workload in production.
In our tests, DEX mode reduced ResNet50 compilation time by 30X.

nGraph further tries to speed up the computation by leveraging
multi-threading and graph scheduling libraries such as OpenMP and
TBB Flow Graph.

Features
--------

nGraph performs a combination of device-specific and
non-device-specific optimizations:

-   **Fusion** -- Fuse multiple ops to to decrease memory usage.
-   **Data layout abstraction** -- Make abstraction easier and faster
    with nGraph translating element order to work best for a given or
    available device.
-   **Data reuse** -- Save results and reuse for subgraphs with the
    same input.
-   **Graph scheduling** -- Run similar subgraphs in parallel via
    multi-threading.
-   **Graph partitioning** -- Partition subgraphs to run on different
    devices to speed up computation; make better use of spare CPU cycles
    with nGraph.
-   **Memory management** -- Prevent peak memory usage by intercepting
    a graph with or by a "saved checkpoint," and to enable data auditing.

Limitations
-----------

The Beta release of nGraph only supports Just-In-Time (JiT) compilation; 
Ahead-of Time (AoT) compilation will be supported in the official release. 
nGraph currently has limited support for dynamic shapes.


Current nGraph Compiler full stack
----------------------------------

![](doc/sphinx/source/graphics/ngraph_full_stack_diagrams.png)

