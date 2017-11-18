#include <iostream>

#include "types.hpp"
#include "nodes/VectorAddNode.hpp"
#include "nodes/InputNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "ImplementationStrategyFactory.hpp"

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
    ConstDataBuffer expected { 5,5,5,5, 4,0,4,0, 1,2,3,4, 0,0,0,0, 2,2,2,-10 };

    // provide input data dimensions
    MemoryDimensions dims({n, dim});
    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("x", dims);
    inputDimensions.emplace("y", dims);

    // set up graph compilation context and platform
    ImplementationStrategyFactory fact;
    CompilationMemoryMap CompilationMemoryMap(inputDimensions);
    std::unique_ptr<GraphCompilationPlatform> platform = fact.CreateGraphCompilationTargetStrategy(CompilationMemoryMap);

    // set up working memory for input nodes (will usually be done during compilation if whole graph is compiled; testing only single node here)
    CompilationMemoryMap.RegisterNodeMemory(&i1, dims);
    CompilationMemoryMap.RegisterNodeMemory(&i2, dims);
    testAddNode.GetMemoryDimensions(CompilationMemoryMap);

    platform->AllocateMemory(&i1);
    platform->AllocateMemory(&i2);
    platform->AllocateMemory(&testAddNode);

    // compile kernel for VectorAddNode object
    testAddNode.Compile(*platform);

    // copy input data into node working memory (will usually be done by compiled kernels for InputNode if whole graph is run; testing only single node here)
    platform->CopyInputData(&i1, input1);
    platform->CopyInputData(&i2, input2);

    // run compiled kernel
    platform->Evaluate();

    // get output (pointer to working memory of VectorAddNode which holds the computation result)
    DataBuffer result(size);
    platform->CopyOutputData(&testAddNode, result);
    //context.GetNodeData(&testAddNode, result);

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
