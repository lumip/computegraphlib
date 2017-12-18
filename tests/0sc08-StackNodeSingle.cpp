#include <iostream>
#include <sstream>
#include <cmath>
#include <memory>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "nodes/StackNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"
#include "GraphCompilationPlatformFactory.hpp"

float testStackNode(const MemoryDimensions sliceDim, const std::vector<DataBuffer> slices, const size_t axis, const MemoryDimensions expectedDim, ConstDataBuffer& expected)
{
    const size_t sliceCount = slices.size();

    // Create, and set up InputNodes holding the slices data
    std::vector<std::unique_ptr<Node>> createdNodes(sliceCount); // keep track of dynamically allocated nodes so that they get freed upon returning
    std::vector<NodePtr> sliceInputs(sliceCount); // vector holding plain pointers to feed into StackNode
    GraphCompilationPlatformFactory fact;
    CompilationMemoryMap compilationMemoryMap;
    std::unique_ptr<GraphCompilationPlatform> platform = fact.CreateGraphCompilationTargetStrategy(compilationMemoryMap);

    for (size_t i = 0; i < sliceCount; ++i)
    {
        std::stringstream ss;
        ss << i;
        NodePtr node = new InputNode(ss.str());
        createdNodes[i] = std::unique_ptr<Node>(node);
        sliceInputs[i] = node;
        compilationMemoryMap.RegisterNodeMemory(node, sliceDim);
        platform->ReserveMemoryBuffer(node);
        node->Compile(*platform);
    }

    // create, set up and compile StackNode
    StackNode stackNode(sliceInputs, axis);
    stackNode.GetMemoryDimensions(compilationMemoryMap);
    platform->ReserveMemoryBuffer(&stackNode);
    platform->AllocateAllMemory();

    stackNode.Compile(*platform);

    // copy input data
    for (size_t i = 0; i < sliceCount; ++i)
    {
        platform->CopyInputData(sliceInputs[i], slices[i].data());
    }

    // run compiled kernel
    platform->Evaluate();

    // get output (pointer to working memory of StackNode which holds the computation result)
    const MemoryDimensions resultDim = compilationMemoryMap.GetNodeMemoryDimensions(&stackNode);
    DataBuffer result(resultDim.size());
    platform->CopyOutputData(&stackNode, result.data());

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
    MemoryDimensions sliceDim, expectedDim;

    float totalError = 0.0f;
    float error = 0.0f;

    std::vector<DataBuffer> sliceData { {1,2,3,4}, {2,3,4,5}, {3,4,5,6}, {4,5,6,7}, {5,6,7,8} }; // 5 times 4
    DataBuffer expected;

    // row slices stacked along y-axis (axis == 0)
    sliceDim = {1, 4};
    expectedDim = {5, 4};
    expected = { 1,2,3,4, 2,3,4,5, 3,4,5,6, 4,5,6,7, 5,6,7,8 };
    error = testStackNode(sliceDim, sliceData, 0, expectedDim, expected);
    std::cout << "Stacking row vector along y-axis | Error: " << error << std::endl;
    totalError += error;

    // row slices stacked along x-axis (axis == 1)
    sliceDim = {1, 4};
    expectedDim = {1, 20};
    expected = { 1,2,3,4, 2,3,4,5, 3,4,5,6, 4,5,6,7, 5,6,7,8 };
    error = testStackNode(sliceDim, sliceData, 1, expectedDim, expected);
    std::cout << "Stacking row vector along x-axis | Error: " << error << std::endl;
    totalError += error;

    // column slices stacked along y-axis (axis == 0)
    sliceDim = {4, 1};
    expectedDim = {20, 1};
    expected = { 1,2,3,4, 2,3,4,5, 3,4,5,6, 4,5,6,7, 5,6,7,8 };
    error = testStackNode(sliceDim, sliceData, 0, expectedDim, expected);
    std::cout << "Stacking column vector along y-axis | Error: " << error << std::endl;
    totalError += error;

    // column slices stacked along x-axis (axis == 1)
    sliceDim = {4, 1};
    expectedDim = {4, 5};
    expected = { 1,2,3,4,5, 2,3,4,5,6, 3,4,5,6,7, 4,5,6,7,8 };
    error = testStackNode(sliceDim, sliceData, 1, expectedDim, expected);
    std::cout << "Stacking column vector along x-axis | Error: " << error << std::endl;
    totalError += error;


    // return 0 if error below threshold, -1 otherwise
    if (totalError < 0.00001)
    {
        return 0;
    }
    return -1;
}
