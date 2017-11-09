#include <iostream>

#include "types.hpp"
#include "nodes/VectorAddNode.hpp"
#include "nodes/InputNode.hpp"
#include "GraphCompilationContext.hpp"

int main(int argc, const char * const argv[])
{
    size_t dim = 4;
    size_t n = 5;
    size_t size = dim * n;

    InputNode i1("x", dim);
    InputNode i2("y", dim);
    VectorAddNode testAddNode(&i1, &i2);

    // define input and expected output data
    InputDataBuffer input1 { 1,2,3,4, 2,2,2,2, 0,0,0,0, 1,0,-1,0, -1,-3,-5,-7 };
    InputDataBuffer input2 { 4,3,2,1, 2,-2,2,-2, 1,2,3,4, -1,0,1,0, 3,5,7,-3 };
    InputDataBuffer expected { 5,5,5,5, 4,0,4,0, 1,2,3,4, 0,0,0,0, 2,2,2,-10 };

    // populate input data mapping
    InputDataMap inputs;
    inputs.emplace("x", std::ref(input1));
    inputs.emplace("y", std::ref(input2));

    // set up graph compilation context
    GraphCompilationContext context(inputs);
    // set up working memory for input nodes (will usually be done during compilation if whole graph is compiled; testing only single node here)
    context.AssignNodeMemory(&i1, context.RegisterMemory(n, dim));
    context.AssignNodeMemory(&i2, context.RegisterMemory(n, dim));
    // compile kernel for VectorAddNode object
    testAddNode.Compile(context);

    // copy input data into node working memory (will usually be done by compiled kernels for InputNode if whole graph is run; testing only single node here)
    float* const memInA = context.GetNodeMemoryDescriptor(&i1).handle;
    float* const memInB = context.GetNodeMemoryDescriptor(&i2).handle;
    std::copy(input1.cbegin(), input1.cend(), memInA);
    std::copy(input2.cbegin(), input2.cend(), memInB);

    // run compiled kernel
    context.Evaluate();

    // get output (pointer to working memory of VectorAddNode which holds the computation result)
    float* const result = context.GetNodeMemoryDescriptor(&testAddNode).handle;

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
}
