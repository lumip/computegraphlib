#include <iostream>
#include <cmath>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "nodes/ReduceMeanNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"
#include "ImplementationStrategyFactory.hpp"

float testReduceMeanNode(const MemoryDimensions input1Dim, const InputDataBuffer& input1, const size_t axis, const MemoryDimensions expectedDim, ConstDataBuffer& expected)
{
    InputNode i1("x", input1Dim.xDim);
    ReduceMeanNode reduceMeanNode(&i1, axis);

    // create InputDimensionsMap object to provide input dimensions to graph compilation routines
    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("x", input1Dim);

    // set up graph compilation context and platform
    ImplementationStrategyFactory fact;
    CompilationMemoryMap compilationMemoryMap(inputDimensions);
    std::unique_ptr<GraphCompilationPlatform> platform = fact.CreateGraphCompilationTargetStrategy(compilationMemoryMap);

    // set up working memory for input nodes (will usually be done during compilation if whole graph is compiled; testing only single node here)
    compilationMemoryMap.RegisterNodeMemory(&i1, input1Dim);
    reduceMeanNode.GetMemoryDimensions(compilationMemoryMap);

    platform->ReserveMemoryBuffer(&i1);
    platform->ReserveMemoryBuffer(&reduceMeanNode);
    platform->AllocateAllMemory();

    // compile kernel for ReduceMeanNode object
    reduceMeanNode.Compile(*platform);

    // copy input data into node working memory (will usually be done by compiled kernels for InputNode if whole graph is run; testing only single node here)
    platform->CopyInputData(&i1, input1);

    // run compiled kernel
    platform->Evaluate();

    // get output (pointer to working memory of ReduceMeanNode which holds the computation result)
    const MemoryDimensions resultDim = compilationMemoryMap.GetNodeMemoryDimensions(&reduceMeanNode);
    DataBuffer result(resultDim.size());
    platform->CopyOutputData(&reduceMeanNode, result);

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

    InputDataBuffer input1 { 1,2,3,4, 2,2,2,2, 0,0,0,0, 1,0,-1,0, -1,-3,-5,-7 }; // 5 x 4

    // y-axis (axis == 0)
    DataBuffer expected { (input1[0]+input1[4]+input1[8]+ input1[12]+input1[16])/5.0f,
                          (input1[1]+input1[5]+input1[9]+ input1[13]+input1[17])/5.0f,
                          (input1[2]+input1[6]+input1[10]+input1[14]+input1[18])/5.0f,
                          (input1[3]+input1[7]+input1[11]+input1[15]+input1[19])/5.0f };

    error = testReduceMeanNode(MemoryDimensions({m, n}), input1, 0, MemoryDimensions({1, n}), expected);
    std::cout << "Reduce along y-axis | Error: " << error << std::endl;
    totalError += error;

    // x-axis (axis == 1)
    expected = { (input1[0] +input1[1] +input1[2] +input1[3] )/4.0f,
                 (input1[4] +input1[5] +input1[6] +input1[7] )/4.0f,
                 (input1[8] +input1[9] +input1[10]+input1[11])/4.0f,
                 (input1[12]+input1[13]+input1[14]+input1[15])/4.0f,
                 (input1[16]+input1[17]+input1[18]+input1[19])/4.0f };

    error = testReduceMeanNode(MemoryDimensions({m, n}), input1, 1, MemoryDimensions({m, 1}), expected);
    std::cout << "Reduce along x-axis | Error: " << error << std::endl;
    totalError += error;

    // return 0 if error below threshold, -1 otherwise
    if (totalError < 0.00001)
    {
        return 0;
    }
    return -1;
}
