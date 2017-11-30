#include <iostream>
#include <algorithm>
#include <random>

#include <mnist/mnist_reader.hpp>

#include "types.hpp"
#include "nodes/nodes.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompiler.hpp"
#include "CompiledGraph.hpp"
#include "ImplementationStrategyFactory.hpp"

std::vector<std::unique_ptr<Node>> nodes;

int main(int argc, const char * const argv[])
{
    const size_t InputDim = 784;
    const size_t BatchSize = 500;
    const size_t OutputDim = 10;


    // build feedforward network for classification
    // declare inputs and variables
    InputNode inputBatch("ImgBatch", InputDim); // BatchSize x InputDim (batch of row vectors of inputs)
    VariableNode weights("Weights", InputDim, OutputDim); // InputDim x OutputDim (weight matrix for multiplication from the right)
    VariableNode bias("Bias", 1, OutputDim); // 1 x OutputDim (row vector of biases)

    // weighting inputs and applying bias
    MatrixMultNode weightedInputs(&inputBatch, &weights); // BatchSize x OutputDim
    VectorAddNode weightedBiasedInputs(&weightedInputs, &bias); // todo: modifiy so that it works with matrix + vector input (tricks with mod); potentially add scalar factor to multiply to second arg

    // apply softmax
    ExpFuncNode expF(&weightedBiasedInputs); // BatchSize x OutputDim
    ReduceSumNode expFSum(&expF, 1); // BatchSize x 1
    VectorDivNode softmax(&expF, &expFSum); // BatchSize x OutputDim

    // add loss function graph part for training
    // compute loss (cross-entropy)
    InputNode correctClasses("Classes", OutputDim); // BatchSize x OutputDim, input for correct results (represented as one-hot vector (correct class indicated by 1 in that dimension))
    LogFuncNode logSoftmax(&softmax); // BatchSize x OutputDim
    VectorMultNode vm(&correctClasses, &logSoftmax); // BatchSize x OutputDim
    ReduceSumNode negCrossEntropy(&vm, 1); // BatchSize x 1
    NegateNode crossEntropy(&negCrossEntropy); // cross entropy per input

    ReduceMeanNode loss(&crossEntropy, 0); // 1x1, mean cross entropy over batch

    // build backpropagation/derivation graph (todo: it would be nice if we could automate this later on)
    NegateNode negCorrectClasses(&correctClasses); // BatchSize x OutputDim
    VectorAddNode gradFactors(&softmax, &negCorrectClasses); // BatchSize x OutputDim
    ReduceMeanNode biasGrad(&gradFactors, 0); // 1 x OutputDim
    std::vector<NodePtr> weightGradComponents(OutputDim);
    for (size_t i = 0; i < OutputDim; ++i)
    {
        NodePtr slice = new SliceNode(&gradFactors, i, 1); // slice i from dim 1, BatchSize x 1
        NodePtr tmp = new VectorMultNode(slice, &inputBatch); // BatchSize x InputDim
        NodePtr reduced = new ReduceMeanNode(tmp, 0); // 1 x InputDim
        nodes.emplace_back(slice); // keep track of nodes so that they are disposed correctly
        nodes.emplace_back(tmp);
        nodes.emplace_back(reduced);
        weightGradComponents[i] = reduced;
    }
    StackNode weightGradTransp(weightGradComponents, 0); // OutputDim x InputDim
    TransposeNode weightGrad(&weightGradTransp); // InputDim x OutputDim

    const float LearningRate = 0.5f;
    ConstMultNode biasUpdateDelta(&biasGrad, LearningRate); // 1 x OutputDim
    VectorAddNode biasUpdated(&bias, &biasUpdateDelta); // 1 x OutputDim
    bias.SetInput(&biasUpdated);
    ConstMultNode weightsUpdateDelta(&weightGrad, LearningRate); // InputDim x OutputDim
    VectorAddNode weightsUpdated(&weights, &weightsUpdateDelta); // InputDim x OutputDim
    weights.SetInput(&weightsUpdated);

    // load inputs and initialize variables
    DataBuffer weightsData(BatchSize * OutputDim);
    DataBuffer biasData(OutputDim);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::generate(std::begin(weightsData), std::end(weightsData), [&dist, &gen]()->float {return dist(gen);});
    std::generate(std::begin(biasData), std::end(biasData), [&dist, &gen]()->float {return dist(gen);});
    std::cout << "done generating input data" << std::endl;

    const std::string dataLocation(MNIST_DATA_DIR);
    mnist::MNIST_dataset<std::vector, DataBuffer, float> dataset = mnist::read_dataset_direct<std::vector, DataBuffer, float>(dataLocation, 0, 0);
    InputDataBuffer& buff(dataset.training_images[0]);
    std::cout << buff.size() << std::endl;

    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("ImgBatch", MemoryDimensions({BatchSize, InputDim}));
    inputDimensions.emplace("Weights", MemoryDimensions({InputDim, OutputDim}));
    inputDimensions.emplace("Bias", MemoryDimensions({1, OutputDim}));
    inputDimensions.emplace("Classes", MemoryDimensions({BatchSize, OutputDim}));

    GraphCompiler compiler(std::unique_ptr<const ImplementationStrategyFactory>(new ImplementationStrategyFactory));
    const std::unique_ptr<CompiledGraph> graph = compiler.Compile(&loss, inputDimensions);
    graph->Evaluate(InputDataMap());

    return 0;
}
