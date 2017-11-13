#include <iostream>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "GraphCompilationContext.hpp"
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

    // set up graph compilation context
    GraphCompilationContext context(inputDimensions, ImplementationStrategyFactory().CreateGraphCompilationTargetStrategy());
    // compile kernel for VectorAddNode object
    testInputNode.Compile(context);

    // prepare input data
    InputDataMap inputs;
    inputs.emplace("x", input1);

    // run compiled kernel
    context.Evaluate(inputs);

    // get output (pointer to working memory of VectorAddNode which holds the computation result)
    DataBuffer result;
    context.GetNodeData(&testInputNode, result);

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
