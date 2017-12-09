#include <iostream>
#include <cmath>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "nodes/TransposeNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"
#include "ImplementationStrategyFactory.hpp"

int main(int argc, const char * const argv[])
{
    size_t m = 5;
    size_t n = 4;

    InputNode i1("x", n);
    TransposeNode transposeNode(&i1);

    // define input and expected output data
    InputDataBuffer input1 { 1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16, 17,18,19,20 }; // 5 x 4
    ConstDataBuffer expected { 1,5, 9,13,17,
                               2,6,10,14,18,
                               3,7,11,15,19,
                               4,8,12,16,20 };
    const MemoryDimensions expectedDim {n, m};

    // set up graph compilation context and platform
    ImplementationStrategyFactory fact;
    CompilationMemoryMap compilationMemoryMap;
    std::unique_ptr<GraphCompilationPlatform> platform = fact.CreateGraphCompilationTargetStrategy(compilationMemoryMap);

    // set up working memory for input nodes (will usually be done during compilation if whole graph is compiled; testing only single node here)
    compilationMemoryMap.RegisterNodeMemory(&i1, {m, n});
    transposeNode.GetMemoryDimensions(compilationMemoryMap);

    platform->ReserveMemoryBuffer(&i1);
    platform->ReserveMemoryBuffer(&transposeNode);
    platform->AllocateAllMemory();

    // compile kernel for TransposeNode object
    transposeNode.Compile(*platform);

    // copy input data into node working memory (will usually be done by compiled kernels for InputNode if whole graph is run; testing only single node here)
    platform->CopyInputData(&i1, input1);

    // run compiled kernel
    platform->Evaluate();

    // get output (pointer to working memory of TransposeNode which holds the computation result)
    const MemoryDimensions resultDim = compilationMemoryMap.GetNodeMemoryDimensions(&transposeNode);
    DataBuffer result(resultDim.size());
    platform->CopyOutputData(&transposeNode, result);

    // compute and output squared error
    float error = 0.0f;
    for (size_t i = 0; i < result.size(); ++i)
    {
        error += std::pow(result[i] - expected[i], 2);
    }
    error += std::pow(static_cast<float>(resultDim.xDim) - static_cast<float>(expectedDim.xDim), 2);
    error += std::pow(static_cast<float>(resultDim.yDim) - static_cast<float>(expectedDim.yDim), 2);

    std::cout << "Error: " << error << std::endl;

    // return 0 if error below threshold, -1 otherwise
    if (error < 0.00001)
    {
        return 0;
    }
    return -1;
}
