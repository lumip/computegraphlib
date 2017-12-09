#include <iostream>
#include <cmath>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "nodes/SliceNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"
#include "ImplementationStrategyFactory.hpp"

float testSliceNode(const MemoryDimensions input1Dim, const InputDataBuffer& input1, const size_t sliceId, const size_t axis, const MemoryDimensions expectedDim, ConstDataBuffer& expected)
{
    InputNode i1("x");
    SliceNode sliceNode(&i1, sliceId, axis);

    // create InputDimensionsMap object to provide input dimensions to graph compilation routines
    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("x", input1Dim);

    // set up graph compilation context and platform
    ImplementationStrategyFactory fact;
    CompilationMemoryMap compilationMemoryMap(inputDimensions);
    std::unique_ptr<GraphCompilationPlatform> platform = fact.CreateGraphCompilationTargetStrategy(compilationMemoryMap);

    // set up working memory for input nodes (will usually be done during compilation if whole graph is compiled; testing only single node here)
    compilationMemoryMap.RegisterNodeMemory(&i1, input1Dim);
    sliceNode.GetMemoryDimensions(compilationMemoryMap);

    platform->ReserveMemoryBuffer(&i1);
    platform->ReserveMemoryBuffer(&sliceNode);
    platform->AllocateAllMemory();

    // compile kernel for SliceNode object
    sliceNode.Compile(*platform);

    // copy input data into node working memory (will usually be done by compiled kernels for InputNode if whole graph is run; testing only single node here)
    platform->CopyInputData(&i1, input1);

    // run compiled kernel
    platform->Evaluate();

    // get output (pointer to working memory of SliceNode which holds the computation result)
    const MemoryDimensions resultDim = compilationMemoryMap.GetNodeMemoryDimensions(&sliceNode);
    DataBuffer result(resultDim.size());
    platform->CopyOutputData(&sliceNode, result);

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
    const MemoryDimensions dim {m, n};
    MemoryDimensions sliceDim;

    float totalError = 0.0f;
    float error = 0.0f;

    InputDataBuffer input1 { 1,2,3,4, 2,2,2,2, 0,0,0,0, 1,0,-1,0, -1,-3,-5,-7 }; // 5 x 4
    DataBuffer expected;

    // x-axis (axis == 1)
    error = 0.0f;
    sliceDim = { m, 1 };
    expected = { input1[0], input1[4], input1[8], input1[12], input1[16] };
    error += testSliceNode(dim, input1, 0, 1, sliceDim, expected);
    expected = { input1[1], input1[5], input1[9], input1[13], input1[17] };
    error += testSliceNode(dim, input1, 1, 1, sliceDim, expected);
    expected = { input1[2], input1[6], input1[10], input1[14], input1[18] };
    error += testSliceNode(dim, input1, 2, 1, sliceDim, expected);
    expected = { input1[3], input1[7], input1[11], input1[15], input1[19] };
    error += testSliceNode(dim, input1, 3, 1, sliceDim, expected);

    std::cout << "Slicing along x-axis (get columns) | Error: " << error << std::endl;
    totalError += error;

    // y-axis (axis == 0)
    error = 0.0f;
    sliceDim = { 1, n };
    expected = { input1[0], input1[1], input1[2], input1[3] };
    error += testSliceNode(dim, input1, 0, 0, sliceDim, expected);
    expected = { input1[4], input1[5], input1[6], input1[7] };
    error += testSliceNode(dim, input1, 1, 0, sliceDim, expected);
    expected = { input1[8], input1[9], input1[10], input1[11] };
    error += testSliceNode(dim, input1, 2, 0, sliceDim, expected);
    expected = { input1[12], input1[13], input1[14], input1[15] };
    error += testSliceNode(dim, input1, 3, 0, sliceDim, expected);
    expected = { input1[16], input1[17], input1[18], input1[19] };
    error += testSliceNode(dim, input1, 4, 0, sliceDim, expected);

    std::cout << "Slicing along y-axis (get rows) | Error: " << error << std::endl;
    totalError += error;

    // return 0 if error below threshold, -1 otherwise
    if (totalError < 0.00001)
    {
        return 0;
    }
    return -1;
}
