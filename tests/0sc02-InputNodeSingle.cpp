#include <iostream>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"
#include "ImplementationStrategyFactory.hpp"

int main(const int argc, const char * const argv[])
{
    size_t dim = 4;
    size_t n = 5;
    size_t size = dim * n;

    InputNode testInputNode("x", dim);

    // define input data
    InputDataBuffer input1 { 1,2,3,4, 2,2,2,2, 0,0,0,0, 1,0,-1,0, -1,-3,-5,-7 };
    InputDataBuffer& expected(input1);

    // provide input data dimensions
    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("x", MemoryDimensions({n, dim}));

    // set up graph compilation context and platform
    ImplementationStrategyFactory fact;
    CompilationMemoryMap CompilationMemoryMap(inputDimensions);
    std::unique_ptr<GraphCompilationPlatform> platform = fact.CreateGraphCompilationTargetStrategy(CompilationMemoryMap);

    // set up working memory for input nodes (will usually be done during compilation if whole graph is compiled; testing only single node here)
    testInputNode.GetMemoryDimensions(CompilationMemoryMap);

    platform->AllocateMemory(&testInputNode);

    // compile kernel for InputNode object
    testInputNode.Compile(*platform);

    platform->CopyInputData(&testInputNode, input1);

    // prepare input data
    InputDataMap inputs;
    inputs.emplace("x", input1);

    // run compiled kernel
    platform->Evaluate(); // todo: currently InputNodes do nothing, not even copying the data.. should probably change

    // get output (pointer to working memory of VectorAddNode which holds the computation result)
    DataBuffer result(size);
    platform->CopyOutputData(&testInputNode, result);
    //context.GetNodeData(&testInputNode, result);

    // compute and output squared error
    float error = 0.0f;
    for (size_t i = 0; i < size; ++i)
    {
        error += (result[i] - expected[i]) * (result[i] - expected[i]);
    }
    std::cout << "Error: " << error << std::endl;

    // return 0 if error below threshold, -1 otherwise
    if (error < 0.00001)
    {
        return 0;
    }
    return -1;

    std::cout << "hello world" << std::endl;
    return 0;
}