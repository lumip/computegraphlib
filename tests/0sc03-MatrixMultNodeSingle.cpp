#include <iostream>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "nodes/MatrixMultNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"
#include "ImplementationStrategyFactory.hpp"

int main(int argc, const char * const argv[])
{
    size_t m = 5;
    size_t n = 4;
    size_t size = m * n;

    InputNode i1("x", n);
    InputNode i2("y", m);
    MatrixMultNode testMultNode(&i1, &i2);

    // define input and expected output data
    InputDataBuffer input1 { 1,2,3,4, 2,2,2,2, 0,0,0,0, 1,0,-1,0, -1,-3,-5,-7 }; // 5 x 4
    InputDataBuffer input2 { 4,3,2,1,2, -2,2,-2,1,2, 3,4,-1,0,1, 0,3,5,7,-3 }; // 4 x 5
    ConstDataBuffer expected { 9,31,15,31,-3, 10,24,8,18,4, 0,0,0,0,0, 1,-1,3,1,1, -13,-50,-26,-53,8 };

    MemoryDimensions dims1 {m, n};
    MemoryDimensions dims2 {n, m};
    MemoryDimensions dimsR {m, m};

    // provide input data dimensions
    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("x", dims1);
    inputDimensions.emplace("y", dims2);

    // set up graph compilation context and platform
    ImplementationStrategyFactory fact;
    CompilationMemoryMap CompilationMemoryMap(inputDimensions);
    std::unique_ptr<GraphCompilationPlatform> platform = fact.CreateGraphCompilationTargetStrategy(CompilationMemoryMap);

    // set up working memory for input nodes (will usually be done during compilation if whole graph is compiled; testing only single node here)
    CompilationMemoryMap.RegisterNodeMemory(&i1, dims1);
    CompilationMemoryMap.RegisterNodeMemory(&i2, dims2);
    testMultNode.GetMemoryDimensions(CompilationMemoryMap);

    platform->AllocateMemory(&i1);
    platform->AllocateMemory(&i2);
    platform->AllocateMemory(&testMultNode);

    // compile kernel for VectorAddNode object
    testMultNode.Compile(*platform);

    // copy input data into node working memory (will usually be done by compiled kernels for InputNode if whole graph is run; testing only single node here)
    platform->CopyInputData(&i1, input1);
    platform->CopyInputData(&i2, input2);

    // run compiled kernel
    platform->Evaluate();

    // get output (pointer to working memory of VectorAddNode which holds the computation result)
    DataBuffer result(dimsR.size());
    platform->CopyOutputData(&testMultNode, result);

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