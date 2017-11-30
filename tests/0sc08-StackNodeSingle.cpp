#include <iostream>
#include <sstream>
#include <cmath>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "nodes/StackNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"
#include "ImplementationStrategyFactory.hpp"

float testStackNode(const MemoryDimensions sliceDim, const std::vector<DataBuffer> slices, const size_t axis, ConstDataBuffer& expected, const MemoryDimensions expectedDim)
{
    const size_t sliceCount = slices.size();

    // Create, and set up InputNodes holding the slices data
    std::vector<NodePtr> sliceInputs(sliceCount);
    ImplementationStrategyFactory fact;
    CompilationMemoryMap compilationMemoryMap;
    std::unique_ptr<GraphCompilationPlatform> platform = fact.CreateGraphCompilationTargetStrategy(compilationMemoryMap);

    for (size_t i = 0; i < sliceCount; ++i)
    {
        std::stringstream ss;
        ss << i;
        NodePtr node = new InputNode(ss.str(), sliceDim.xDim);
        sliceInputs[i] = node;
        compilationMemoryMap.RegisterNodeMemory(node, sliceDim);
        platform->AllocateMemory(node);
        platform->CopyInputData(node, slices[i]);
    }

    // create, set up and compile StackNode
    StackNode stackNode(sliceInputs, axis);
    stackNode.GetMemoryDimensions(compilationMemoryMap);
    platform->AllocateMemory(&stackNode);
    stackNode.Compile(*platform);

    // run compiled kernel
    platform->Evaluate();

    // get output (pointer to working memory of StackNode which holds the computation result)
    DataBuffer result(compilationMemoryMap.GetNodeMemoryDimensions(&stackNode).size());
    platform->CopyOutputData(&stackNode, result);

    // compute and return squared error
    float error = 0.0f;
    for (size_t i = 0; i < result.size(); ++i)
    {
        error += std::pow(result[i] - expected[i], 2);
    }
    const MemoryDimensions resultDim = compilationMemoryMap.GetNodeMemoryDimensions(&stackNode);
    error += std::pow(resultDim.xDim - expectedDim.xDim, 2) + std::pow(resultDim.yDim - expectedDim.yDim, 2);
    return error;
}

int main(int argc, const char * const argv[])
{
    MemoryDimensions sliceDim, expectedDim;

    float totalError = 0.0f;
    float error = 0.0f;

    std::vector<DataBuffer> sliceData { {1,2,3,4}, {2,3,4,5}, {3,4,5,6}, {4,5,6,7}, {5,6,7,8} }; // 5 times 4
    DataBuffer expected;

    // row slices stacked along y-axis (axis == 0)
    sliceDim = {1, 4};
    expectedDim = {5, 4};
    expected = { 1,2,3,4, 2,3,4,5, 3,4,5,6, 4,5,6,7, 5,6,7,8 };
    error = testStackNode(sliceDim, sliceData, 0, expected, expectedDim);
    std::cout << "Stacking row vector along y-axis | Error: " << error << std::endl;
    totalError += error;

    // row slices stacked along x-axis (axis == 1)
    sliceDim = {1, 4};
    expectedDim = {1, 20};
    expected = { 1,2,3,4, 2,3,4,5, 3,4,5,6, 4,5,6,7, 5,6,7,8 };
    error = testStackNode(sliceDim, sliceData, 1, expected, expectedDim);
    std::cout << "Stacking row vector along x-axis | Error: " << error << std::endl;
    totalError += error;

    // column slices stacked along y-axis (axis == 0)
    sliceDim = {4, 1};
    expectedDim = {20, 1};
    expected = { 1,2,3,4, 2,3,4,5, 3,4,5,6, 4,5,6,7, 5,6,7,8 };
    error = testStackNode(sliceDim, sliceData, 0, expected, expectedDim);
    std::cout << "Stacking column vector along y-axis | Error: " << error << std::endl;
    totalError += error;

    // column slices stacked along x-axis (axis == 1)
    sliceDim = {4, 1};
    expectedDim = {4, 5};
    expected = { 1,2,3,4,5, 2,3,4,5,6, 3,4,5,6,7, 4,5,6,7,8 };
    error = testStackNode(sliceDim, sliceData, 1, expected, expectedDim);
    std::cout << "Stacking column vector along x-axis | Error: " << error << std::endl;
    totalError += error;


    // return 0 if error below threshold, -1 otherwise
    if (totalError < 0.00001)
    {
        return 0;
    }
    return -1;
}
