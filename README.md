# Compute Graph Framework

An experimental implementation of a compute graph framework similar to TensorFlow using C++ and OpenCL.

Implemented during a term project for the course Multicore Programming Fundamentals during the winter semester 2017 at Yonsei University.

Applications using the framework can represent arbitrary computations on data as hardware-agnostic computation graphs using the *Node* classes exposed by the framework API. The framework is then able to compile that graph for a specific platform on which it can then be executed on specific input data.

Currently, all node operations are implemented on a sequential CPU and OpenCL/GPU platform.

## Repository Structure
The repository contains the actual framework code in the *cglib* subdirectory as well as a plethora of small tests and examples in the *tests* subdirectory.

The framework code is split into general *includes* and platform-independent code in the *src* directories and the platform-specific implementations in the *src-cpu* and *src-gpu* subdirectories of the *cglib* folder.

All test exectuables are single C++-files and reside directly within the *tests* folder. They adhere to the following naming convention:
```(0s|1g)(c|p)<sequence nr>[b[g]]-<descriptive name>```
- ```0s``` and ```1g``` denote whether the executable tests a *single* node implementation or a full *graph*
- ```c``` and ```p``` denote whether it is a *correctness*/sanity or a *performance* test.
- ```<sequence nr>``` is a sequentially increasing number for ordering the samples
- ```b``` and ```bg``` optionally denote baseline and OpenCL/GPU baseline implementations not making use of the framework

## Compilation
Compilation is driven by *CMake* which will automatically detect *OpenCL* and *PAPI*. Finding *PAPI* might require setting the *PAPI_PREFIX* variable for *CMake* to the installation path of *PAPI*. *PAPI* is not required for compilation of the framework, but the performance test exectuables will not be built if *PAPI* cannot be found.

Building will produce to versions of the library, *cgLibCPU* and *cgLibGPU* as well as all test exectuables linked against both platform-specific libraries, identified with an additional ```c``` or ```g``` to the prefix-code denoting *C*PU or OpenCL/*G*PU code.

To build, create a separate build directory and invoke cmake as follows:
```
mkdir cglib-build && cd cglib-build
cmake [-DPAPI_PREFIX=<papi_directory>] ../computegraphlib
make
```

Make sure to specify Release or Debug build settings during the invocation of *CMake* if required.

## Using the Framework
The following gives a very brief example on how to use the framework to compute ```z = a * (x + y)```.
```
#include "types.hpp"
#include "nodes/nodes.hpp"
#include "GraphCompiler.hpp"
#include "CompiledGraph.hpp"
#include "GraphCompilationPlatformFactory.hpp"

int main(int argc, const char* argv[])
{

    // create graph structure
    InputNode x("x");
    InputNode y("y");
    VariableNode a("a");

    VectorAddNode add(&x, &y);
    VectorMultNode z(&a, &add);

    // specify dimensions of inputs and compile graph to platform
    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("x", MemoryDimensions({1, 5}));
    inputDimensions.emplace("y", MemoryDimensions({1, 5}));
    inputDimensions.emplace("a", MemoryDimensions({1, 1}));

    GraphCompiler compiler(std::unique_ptr<const GraphCompilationPlatformFactory>(new GraphCompilationPlatformFactory));
    const std::unique_ptr<CompiledGraph> graph = compiler.Compile(&z, inputDimensions);

    // in real code: have buffers holding the input data. here just stubs
    float dataA = 2.0f;
    float dataX[5] = { 1.f, 2.f, 3.f, 4.f, 5.f };
    float dataY[5] = { 5.f, 4.f, 3.f, 2.f, 1.f };
    float dataZ[5] = { 0.f, 0.f, 0.f, 0.f, 0.f };

    // initialize variables
    InputDataMap variableMap;
    variableMap.emplace("a", &dataA);
    graph->InitializeVariables(variableMap);

    // run graph on input data
    InputDataMap inputMap;
    inputMap.emplace("x", dataX);
    inputMap.emplace("y", dataY);
    graph->Evaluate(inputMap);

    // get output data
    graph->GetNodeData(&z, dataZ);

    return 0;
}
```
### Selecting the Platform
Selecting the platform is currently done by simply linking against the desired library version of the framework. This will supply a platform-specific implementation of *GraphCompilationPlatformFactory*.

## Code Quality
Unfortunately, the framework code is not the cleanest right now. There are plenty of code duplications, inconsistent use of syntax and unclean responsibilities which are in need of major refactoring. Please don't judge too hard ><.

## Third Party Code
The repository includes files from the MNIST loader published at https://github.com/wichtounet/mnist under the MIT license used in the MNIST classifier example executables. These files are located in *tests/mnist*.