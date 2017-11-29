#include <iostream>

#include "types.hpp"
#include "nodes/VectorAddNode.hpp"
#include "nodes/InputNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"
#include "ImplementationStrategyFactory.hpp"

float testVectorAddNode(MemoryDimensions input1Dim, InputDataBuffer& input1, MemoryDimensions input2Dim, InputDataBuffer& input2, ConstDataBuffer& expected)
{
    InputNode i1("x", input1Dim.xDim);
    InputNode i2("y", input2Dim.xDim);
    VectorAddNode testAddNode(&i1, &i2);

    // create InputDimensionsMap object to provide input dimensions to graph compilation routines
    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("x", input1Dim);
    inputDimensions.emplace("y", input2Dim);

    // set up graph compilation context and platform
    ImplementationStrategyFactory fact;
    CompilationMemoryMap compilationMemoryMap(inputDimensions);
    std::unique_ptr<GraphCompilationPlatform> platform = fact.CreateGraphCompilationTargetStrategy(compilationMemoryMap);

    // set up working memory for input nodes (will usually be done during compilation if whole graph is compiled; testing only single node here)
    compilationMemoryMap.RegisterNodeMemory(&i1, input1Dim);
    compilationMemoryMap.RegisterNodeMemory(&i2, input2Dim);
    testAddNode.GetMemoryDimensions(compilationMemoryMap);

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
    DataBuffer result(compilationMemoryMap.GetNodeMemoryDimensions(&testAddNode).size());
    platform->CopyOutputData(&testAddNode, result);

    // compute and return squared error
    float error = 0.0f;
    for (size_t i = 0; i < result.size(); ++i)
    {
        error += (result[i] - expected[i]) * (result[i] - expected[i]);
    }
    return error;
}

int main(int argc, const char * const argv[])
{
    size_t m = 5;
    size_t n = 4;

    float totalError = 0.0f;
    float error = 0.0f;


    // similar sized data
    DataBuffer input1 { 1,2,3,4, 2,2,2,2, 0,0,0,0, 1,0,-1,0, -1,-3,-5,-7 };
    DataBuffer input2 { 4,3,2,1, 2,-2,2,-2, 1,2,3,4, -1,0,1,0, 3,5,7,-3 };
    DataBuffer expected { 5,5,5,5, 4,0,4,0, 1,2,3,4, 0,0,0,0, 2,2,2,-10 };

    error = testVectorAddNode(MemoryDimensions({m, n}), input1, MemoryDimensions({m, n}), input2, expected);
    std::cout << "Same-size data | Error: " << error << std::endl;
    totalError += error;

    // row broadcasting
    input2 = { 1,2,3,4 };
    expected = { 2,4,6,8, 3,4,5,6, 1,2,3,4, 2,2,2,4, 0,-1,-2,-3 };

    error = testVectorAddNode(MemoryDimensions({m, n}), input1, MemoryDimensions({1, n}), input2, expected);
    std::cout << "Row broadcasting (B) | Error: " << error << std::endl;
    totalError += error;

    error = testVectorAddNode(MemoryDimensions({1, n}), input2, MemoryDimensions({m, n}), input1, expected);
    std::cout << "Row broadcasting (A) | Error: " << error << std::endl;
    totalError += error;

    // column broadcasting
    input2 = { 1, 2, 3, 4, 5 };
    expected = { 2,3,4,5, 4,4,4,4, 3,3,3,3, 5,4,3,4, 4,2,0,-2 };

    error = testVectorAddNode(MemoryDimensions({m, n}), input1, MemoryDimensions({m, 1}), input2, expected);
    std::cout << "Column broadcasting (B) | Error: " << error << std::endl;
    totalError += error;

    error = testVectorAddNode(MemoryDimensions({m, 1}), input2, MemoryDimensions({m, n}), input1, expected);
    std::cout << "Column broadcasting (B) | Error: " << error << std::endl;
    totalError += error;

    // return 0 if error below threshold, -1 otherwise
    if (totalError < 0.00001)
    {
        return 0;
    }
    return -1;
}
