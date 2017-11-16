#include <iostream>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "nodes/MatrixMultNode.hpp"
#include "GraphCompilationContext.hpp"
#include "ImplementationStrategyFactory.hpp"

int main(int argc, const char * const argv[])
{
    size_t dim = 4;
    size_t n = 5;
    size_t size = dim * n;

    InputNode i1("x", 4);
    InputNode i2("y", 5);
    MatrixMultNode testMultNode(&i1, &i2);

    // define input and expected output data
    InputDataBuffer input1 { 1,2,3,4, 2,2,2,2, 0,0,0,0, 1,0,-1,0, -1,-3,-5,-7 }; // 5 x 4
    InputDataBuffer input2 { 4,3,2,1,2, -2,2,-2,1,2, 3,4,-1,0,1, 0,3,5,7,-3 }; // 4 x 5
    ConstDataBuffer expected { 9,31,15,31,-3, 10,24,8,18,4, 0,0,0,0,0, 1,-1,3,1,1, -13,-50,-26,-53,8 };

    // provide input data dimensions
    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("x", MemoryDimensions({n, dim}));
    inputDimensions.emplace("y", MemoryDimensions({dim, n}));

    ImplementationStrategyFactory fact;
    std::unique_ptr<GraphCompilationTargetStrategy> strategy = fact.CreateGraphCompilationTargetStrategy();
    std::unique_ptr<NodeCompiler> nodeCompiler = fact.CreateNodeCompiler(strategy.get());

    // set up graph compilation context
    GraphCompilationContext context(inputDimensions, std::move(strategy));
    // set up working memory for input nodes (will usually be done during compilation if whole graph is compiled; testing only single node here)
    context.AssignNodeMemory(&i1, context.AllocateMemory(inputDimensions.at("x")).handle);
    context.AssignNodeMemory(&i2, context.AllocateMemory(inputDimensions.at("y")).handle);
    // compile kernel for VectorAddNode object
    testMultNode.Compile(context, *nodeCompiler);

    // copy input data into node working memory (will usually be done by compiled kernels for InputNode if whole graph is run; testing only single node here)
    context.SetNodeData(&i1, input1);
    context.SetNodeData(&i2, input2);

    // run compiled kernel
    context.Evaluate(InputDataMap());

    // get output (pointer to working memory of VectorAddNode which holds the computation result)
    DataBuffer result(25);
    std::fill(std::begin(result), std::end(result), 3.3f);
    context.GetNodeData(&testMultNode, result);

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
