#include <iostream>
#include <cmath>

#include "types.hpp"
#include "nodes/VectorDivNode.hpp"
#include "nodes/InputNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"
#include "ImplementationStrategyFactory.hpp"

float testVectorDivNode(const MemoryDimensions input1Dim, InputDataBuffer& input1, const MemoryDimensions input2Dim, InputDataBuffer& input2, const MemoryDimensions expectedDim, ConstDataBuffer& expected)
{
    InputNode i1("x");
    InputNode i2("y");
    VectorDivNode testDivNode(&i1, &i2);

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
    testDivNode.GetMemoryDimensions(compilationMemoryMap);

    platform->ReserveMemoryBuffer(&i1);
    platform->ReserveMemoryBuffer(&i2);
    platform->ReserveMemoryBuffer(&testDivNode);
    platform->AllocateAllMemory();

    // compile kernels
    i1.Compile(*platform);
    i2.Compile(*platform);
    testDivNode.Compile(*platform);

    // copy input data into node working memory (will usually be done by compiled kernels for InputNode if whole graph is run; testing only single node here)
    platform->CopyInputData(&i1, input1);
    platform->CopyInputData(&i2, input2);

    // run compiled kernel
    platform->Evaluate();

    // get output (pointer to working memory of VectorDivNode which holds the computation result)
    const MemoryDimensions resultDim = compilationMemoryMap.GetNodeMemoryDimensions(&testDivNode);
    DataBuffer result(resultDim.size());
    platform->CopyOutputData(&testDivNode, result);

    // compute and return squared error
    float error = 0.0f;
    for (size_t i = 0; i < result.size(); ++i)
    {
        error += std::pow(result[i] - expected[i], 2);
    }
    error += std::pow(static_cast<float>(resultDim.xDim) - static_cast<float>(expectedDim.xDim), 2);
    error += std::pow(static_cast<float>(resultDim.yDim) - static_cast<float>(expectedDim.yDim), 2);
    return error;
}

int main(int argc, const char * const argv[])
{
    size_t m = 5;
    size_t n = 4;

    float totalError = 0.0f;
    float error = 0.0f;


    // similar sized data
    DataBuffer input1 { 1,2,3,4, 2,2,2,2, 0,0,0,0, 1,6,-1,16, -1,-3,-5,-7 };
    DataBuffer input2 { 4,3,2,1, 2,-2,2,-2, 1,2,3,4, -1,3,1,-4, 3,5,7,-3 };
    DataBuffer expected { input1[0]/input2[0], input1[1]/input2[1], input1[2]/input2[2], input1[3]/input2[3],
                          input1[4]/input2[4], input1[5]/input2[5], input1[6]/input2[6], input1[7]/input2[7],
                          input1[8]/input2[8], input1[9]/input2[9], input1[10]/input2[10], input1[11]/input2[11],
                          input1[12]/input2[12], input1[13]/input2[13], input1[14]/input2[14], input1[15]/input2[15],
                          input1[16]/input2[16], input1[17]/input2[17], input1[18]/input2[18], input1[19]/input2[19] };

    error = testVectorDivNode(MemoryDimensions({m, n}), input1, MemoryDimensions({m, n}), input2, MemoryDimensions({m, n}), expected);
    std::cout << "Same-size data | Error: " << error << std::endl;
    totalError += error;

    // row broadcasting
    input2 = { 1,2,3,4 };
    expected = { input1[0]/input2[0], input1[1]/input2[1], input1[2]/input2[2], input1[3]/input2[3],
                 input1[4]/input2[0], input1[5]/input2[1], input1[6]/input2[2], input1[7]/input2[3],
                 input1[8]/input2[0], input1[9]/input2[1], input1[10]/input2[2], input1[11]/input2[3],
                 input1[12]/input2[0], input1[13]/input2[1], input1[14]/input2[2], input1[15]/input2[3],
                 input1[16]/input2[0], input1[17]/input2[1], input1[18]/input2[2], input1[19]/input2[3] };

    error = testVectorDivNode(MemoryDimensions({m, n}), input1, MemoryDimensions({1, n}), input2, MemoryDimensions({m, n}), expected);
    std::cout << "Row broadcasting (B) | Error: " << error << std::endl;
    totalError += error;

    // column broadcasting
    input2 = { 1, 2, 3, 4, 5 };
    expected = { input1[0]/input2[0], input1[1]/input2[0], input1[2]/input2[0], input1[3]/input2[0],
                 input1[4]/input2[1], input1[5]/input2[1], input1[6]/input2[1], input1[7]/input2[1],
                 input1[8]/input2[2], input1[9]/input2[2], input1[10]/input2[2], input1[11]/input2[2],
                 input1[12]/input2[3], input1[13]/input2[3], input1[14]/input2[3], input1[15]/input2[3],
                 input1[16]/input2[4], input1[17]/input2[4], input1[18]/input2[4], input1[19]/input2[4] };

    error = testVectorDivNode(MemoryDimensions({m, n}), input1, MemoryDimensions({m, 1}), input2, MemoryDimensions({m, n}), expected);
    std::cout << "Column broadcasting (B) | Error: " << error << std::endl;
    totalError += error;

    // return 0 if error below threshold, -1 otherwise
    if (totalError < 0.00001)
    {
        return 0;
    }
    return -1;
}
