#include <iostream>
#include <cmath>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "nodes/ExpFuncNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"
#include "ImplementationStrategyFactory.hpp"

int main(int argc, const char * const argv[])
{
    size_t m = 5;
    size_t n = 4;

    InputNode i1("x", n);
    ExpFuncNode expFuncNode(&i1);

    // define input and expected output data
    InputDataBuffer input1 { 1,2,3,4, 2,2,2,2, 0,0,0,0, 1,0,-1,0, -1,-3,-5,-7 };
    ConstDataBuffer expected { std::exp(input1[ 0]),std::exp(input1[ 1]), std::exp(input1[ 2]), std::exp(input1[ 3]),
                               std::exp(input1[ 4]),std::exp(input1[ 5]), std::exp(input1[ 6]), std::exp(input1[ 7]),
                               std::exp(input1[ 8]),std::exp(input1[ 9]), std::exp(input1[10]), std::exp(input1[11]),
                               std::exp(input1[12]),std::exp(input1[13]), std::exp(input1[14]), std::exp(input1[15]),
                               std::exp(input1[16]),std::exp(input1[17]), std::exp(input1[18]), std::exp(input1[19])};
    const MemoryDimensions expectedDim {m, n};

    // set up graph compilation context and platform
    ImplementationStrategyFactory fact;
    CompilationMemoryMap compilationMemoryMap;
    std::unique_ptr<GraphCompilationPlatform> platform = fact.CreateGraphCompilationTargetStrategy(compilationMemoryMap);

    // set up working memory for input nodes (will usually be done during compilation if whole graph is compiled; testing only single node here)
    compilationMemoryMap.RegisterNodeMemory(&i1, {m, n});
    expFuncNode.GetMemoryDimensions(compilationMemoryMap);

    platform->AllocateMemory(&i1);
    platform->AllocateMemory(&expFuncNode);

    // compile kernel for ExpFuncNode object
    expFuncNode.Compile(*platform);

    // copy input data into node working memory (will usually be done by compiled kernels for InputNode if whole graph is run; testing only single node here)
    platform->CopyInputData(&i1, input1);

    // run compiled kernel
    platform->Evaluate();

    // get output (pointer to working memory of ExpFuncNode which holds the computation result)
    const MemoryDimensions resultDim = compilationMemoryMap.GetNodeMemoryDimensions(&expFuncNode);
    DataBuffer result(resultDim.size());
    platform->CopyOutputData(&expFuncNode, result);

    // compute and output squared error
    float error = 0.0f;
    for (size_t i = 0; i < result.size(); ++i)
    {
        error += std::pow(result[i] - expected[i], 2);
    }
    error += std::pow(resultDim.xDim - expectedDim.xDim, 2) + std::pow(resultDim.yDim - expectedDim.yDim, 2);

    std::cout << "Error: " << error << std::endl;

    // return 0 if error below threshold, -1 otherwise
    if (error < 0.00001)
    {
        return 0;
    }
    return -1;
}
