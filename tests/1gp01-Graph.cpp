#include <iostream>
#include <algorithm>
#include <random>
#include <assert.h>

#include <papi.h>

#include "types.hpp"
#include "nodes/nodes.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompiler.hpp"
#include "CompiledGraph.hpp"
#include "ImplementationStrategyFactory.hpp"

int main(int argc, const char * const argv[])
{
    const size_t xDim = 200;
    const size_t yDim = 200;
    const size_t sharedDim = 100;
    const size_t branches = 1 << 13; // number of parallel multiplications

    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if(retval != PAPI_VER_CURRENT)
    {
        std::cout << "could not initialize PAPI" << std::endl;
        return -1;
    }

    // create a graph of thousands of parallel branches matrix multiplication followed by element-wise logarithm and then summing up the results.
    // each single multiplication is rather small, so this tests benefits of parallelizing kernel execution.

    InputNode xT("x");
    InputNode y("y");

    std::vector<std::unique_ptr<Node>> nodes;
    for (size_t i = 0; i < branches; ++i) // creates nodes for all parallel multiplications
    {
        nodes.emplace_back(new MatrixMultNode(&xT, &y));
    }
    size_t nextOffset = nodes.size();
    for (size_t i = 0; i < branches; ++i)
    {
        nodes.emplace_back(new LogFuncNode(nodes[i].get()));
    }
    size_t currentOffset = nextOffset;
    for (size_t l = branches; l > 1; l >>= 1) // sum all computations results in a parallel tree pattern
    {
        assert(l % 2 == 0);
        nextOffset = nodes.size();
        for (size_t i = 0; i < l; i += 2)
        {
            nodes.emplace_back(new VectorAddNode(nodes[currentOffset + i].get(), nodes[currentOffset + i + 1].get()));
        }
        currentOffset = nextOffset;
    }
    ConstNodePtr finalNode = nodes[nodes.size() - 1].get();

    // specify concrete input dimensions for compilation
    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("x", MemoryDimensions({xDim, sharedDim}));
    inputDimensions.emplace("y", MemoryDimensions({sharedDim, yDim}));

    // compile the graph
    long long time_setup_start = PAPI_get_real_nsec();
    GraphCompiler compiler(std::unique_ptr<const ImplementationStrategyFactory>(new ImplementationStrategyFactory));
    const std::unique_ptr<CompiledGraph> graph = compiler.Compile(finalNode, inputDimensions);
    long long time_setup_stop = PAPI_get_real_nsec();

    // prepare inputs
    float* const xData = graph->GetMappedInputBuffer("x");
    float* const yData = graph->GetMappedInputBuffer("y");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::cout << "generating data" << std::endl;
    std::generate(xData, xData + (xDim * sharedDim), [&dist, &gen]()->float {return dist(gen);});
    std::generate(yData, yData + (yDim * sharedDim), [&dist, &gen]()->float {return dist(gen);});

    InputDataMap inputDataMap;
    inputDataMap.emplace("x", xData);
    inputDataMap.emplace("y", yData);

    std::cout << "evaluating graph" << std::endl;
    // evaluate graph
    long long time_start = PAPI_get_real_nsec();
    graph->Evaluate(inputDataMap);
    long long time_stop = PAPI_get_real_nsec();

    std::cout << "Setup: " << time_setup_stop - time_setup_start << " ns; Copy: -1 ns; Compute+Copy: " << time_stop - time_start << " ns" << std::endl;

    return 0;
}
