#include <iostream>
#include <algorithm>
#include <random>
#include <assert.h>

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
    GraphCompiler compiler(std::unique_ptr<const ImplementationStrategyFactory>(new ImplementationStrategyFactory));
    const std::unique_ptr<CompiledGraph> graph = compiler.Compile(finalNode, inputDimensions);

    // prepare inputs
    DataBuffer xData(xDim * sharedDim);
    DataBuffer yData(yDim * sharedDim);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::cout << "generating data" << std::endl;
    std::generate(std::begin(xData), std::end(xData), [&dist, &gen]()->float {return dist(gen);});
    std::generate(std::begin(yData), std::end(yData), [&dist, &gen]()->float {return dist(gen);});

    InputDataMap inputDataMap;
    inputDataMap.emplace("x", xData);
    inputDataMap.emplace("y", yData);

    std::cout << "evaluating graph" << std::endl;
    // evaluate graph
    graph->Evaluate(inputDataMap);

    return 0;
}
