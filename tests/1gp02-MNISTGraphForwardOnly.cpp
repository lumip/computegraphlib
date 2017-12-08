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
    // Check command line arguments
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <path to MNIST dataset>" << std::endl;
        return -1;
    }
    const std::string mnistDataDir(argv[1]);

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
    VectorAddNode weightedBiasedInputs(&weightedInputs, &bias);

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

    // load inputs and initialize variables
    DataBuffer weightsData(InputDim * OutputDim);
    DataBuffer biasData(OutputDim);

    /*std::random_device rd;
    std::mt19937 gen(rd());*/
    std::mt19937 gen(22); // with fixed seed, reproducible results
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::generate(std::begin(weightsData), std::end(weightsData), [&dist, &gen]()->float {return dist(gen);});
    std::generate(std::begin(biasData), std::end(biasData), [&dist, &gen]()->float {return dist(gen);});

    mnist::MNIST_dataset<std::vector, DataBuffer, uint8_t> dataset = mnist::read_dataset_direct<std::vector, DataBuffer, uint8_t>(mnistDataDir, 0, 0);

    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("ImgBatch", MemoryDimensions({BatchSize, InputDim}));
    inputDimensions.emplace("Weights", MemoryDimensions({InputDim, OutputDim}));
    inputDimensions.emplace("Bias", MemoryDimensions({1, OutputDim}));
    inputDimensions.emplace("Classes", MemoryDimensions({BatchSize, OutputDim}));

    GraphCompiler compiler(std::unique_ptr<const ImplementationStrategyFactory>(new ImplementationStrategyFactory));
    const std::unique_ptr<CompiledGraph> graph = compiler.Compile(&loss, inputDimensions);

    DataBuffer imgInputData(BatchSize * InputDim);
    DataBuffer classesInputData(BatchSize * OutputDim);
    std::fill(std::begin(classesInputData), std::begin(classesInputData), 0.0f);
    {
        DataBuffer::iterator it = std::begin(imgInputData);
        for (size_t i = 0; i < BatchSize; ++i)
        {
            InputDataBuffer& sampleImageBuffer = dataset.training_images[i];
            it = std::copy(std::begin(sampleImageBuffer), std::end(sampleImageBuffer), it);
            uint8_t sampleLabel = dataset.training_labels[i];
            classesInputData[i * OutputDim + sampleLabel] = 1.0f;
        }
        for (size_t i = 0; i < imgInputData.size(); ++i)
        {
            imgInputData[i] /= 255.0f;
        }
    }

    InputDataMap variablesDataMap;
    variablesDataMap.emplace("Weights", weightsData);
    variablesDataMap.emplace("Bias", biasData);
    graph->InitializeVariables(variablesDataMap);

    InputDataMap inputDataMap;
    inputDataMap.emplace("ImgBatch", imgInputData);
    inputDataMap.emplace("Classes", classesInputData);

    graph->Evaluate(inputDataMap);

    DataBuffer lossOutput(1);
    graph->GetNodeData(&loss, lossOutput);
    std::cout << lossOutput.size() << std::endl;
    std::cout << "Loss: " << lossOutput[0] << std::endl;

    return 0;
}
