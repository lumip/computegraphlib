#include <iostream>
#include <algorithm>
#include <random>

#include "types.hpp"
#include "nodes/nodes.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompiler.hpp"
#include "CompiledGraph.hpp"
#include "GraphCompilationPlatformFactory.hpp"

int main(int argc, const char * const argv[])
{
    const size_t InputDim = 4;
    const size_t BatchSize = 5;
    const size_t OutputDim = 2;


    // very simple graph which computes: softmax(X*Y+b)
    // declare inputs and variables
    InputNode X("X"); // BatchSize x InputDim
    VariableNode Y("Y"); // InputDim x OutputDim
    VariableNode b("b"); // 1 x OutputDim

    // weighting inputs and applying bias
    MatrixMultNode XY(&X, &Y); // BatchSize x OutputDim
    VectorAddNode XYb(&XY, &b);

    // apply softmax
    ExpFuncNode expXYb(&XYb); // BatchSize x OutputDim
    ReduceSumNode sumexpXYb(&expXYb, 1); // BatchSize x 1
    VectorDivNode softmax(&expXYb, &sumexpXYb); // BatchSize x OutputDim

    // create the exact same network one more time with separate nodes (to test a parallel independent execution path)
    MatrixMultNode XY2(&X, &Y);
    VectorAddNode XYb2(&XY2, &b);
    ExpFuncNode expXYb2(&XYb2);
    ReduceSumNode sumexpXYb2(&expXYb2, 1);
    VectorDivNode softmax2(&expXYb2, &sumexpXYb2);

    TransposeNode softmaxT(&softmax);
    TransposeNode softmax2T(&softmax2);
    VectorAddNode combiner(&softmax, &softmax2); // we are not interested in adding it up but we need a single result node for compilation
                                                 // (preceeding tranposes serve to break up potential buffer reuse of VectorAddNode)

    // specify concrete input dimensions for compilation
    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("X", MemoryDimensions({BatchSize, InputDim}));
    inputDimensions.emplace("Y", MemoryDimensions({InputDim, OutputDim}));
    inputDimensions.emplace("b", MemoryDimensions({1, OutputDim}));

    // compile the graph
    GraphCompiler compiler(std::unique_ptr<const GraphCompilationPlatformFactory>(new GraphCompilationPlatformFactory));
    const std::unique_ptr<CompiledGraph> graph = compiler.Compile(&combiner, inputDimensions);  // we compile combiner node
                                                                                                // but we will ignore it afterwards
                                                                                                // and get results from the sotmax nodes

    // prepare inputs
    DataBuffer dataX = { 1,2,3,4,
                         2,3,4,5,
                         3,4,5,6,
                         4,5,6,7,
                         5,6,7,8 };

    DataBuffer dataY = { 0.5f, 0.5f,
                         0.75f, 0.25f,
                         0.25f, 0.75f,
                         0.9f, 0.1f };

    DataBuffer datab = { 0.33f, -0.33f };

    DataBuffer expected = { 0.96643072f, 0.03356923f,
                            0.98463225f, 0.01536771f,
                            0.99303585f, 0.00696409f,
                            0.99685884f, 0.00314121f,
                            0.99858618f, 0.00141388f };

    // initialize variables
    InputDataMap variableDataMap;
    variableDataMap["Y"] = dataY.data();
    variableDataMap["b"] = datab.data();
    //variableDataMap.emplace("Y", dataY.data());
    //variableDataMap.emplace("b", datab.data());
    graph->InitializeVariables(variableDataMap);

    InputDataMap inputDataMap;
    inputDataMap.emplace("X", dataX.data());

    // evaluate graph
    graph->Evaluate(inputDataMap);

    // otbain result
    DataBuffer result(BatchSize * OutputDim);
    DataBuffer result2(BatchSize * OutputDim);
    graph->GetNodeData(&softmax, result.data());
    graph->GetNodeData(&softmax2, result2.data());

    // compute and return squared error
    float error = 0.0f;
    for (size_t i = 0; i < OutputDim; ++i)
    {
        error += std::pow(result[i] - expected[i], 2);
        error += std::pow(result2[i] - expected[i], 2);
    }
    std::cout << "Error: " << error << std::endl;

    // return 0 if error below threshold, -1 otherwise
    if (error < 0.00001)
    {
        return 0;
    }
    return -1;
}
