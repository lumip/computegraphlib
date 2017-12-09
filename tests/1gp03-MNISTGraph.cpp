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

class MNISTGraph
{
private:
    std::vector<std::unique_ptr<Node>> _nodes;
    InputNode* inputBatch;
    InputNode* correctClasses;
    VariableNode* weights;
    VariableNode* bias;
public:
    MNISTGraph()
        : _nodes()
        , inputBatch(new InputNode("ImgBatch")) // constructor sets up input and variable nodes
        , correctClasses(new InputNode("Classes"))
        , weights(new VariableNode("Weights"))
        , bias(new VariableNode("Bias"))
    {
        _nodes.emplace_back(inputBatch), _nodes.emplace_back(correctClasses);
        _nodes.emplace_back(weights); _nodes.emplace_back(bias);
    }
    ~MNISTGraph() { }
    NodePtr ConstructClassifierGraph()
    {
        // build feedforward network for classification
        // weighting inputs and applying bias
        MatrixMultNode* weightedInputs = new MatrixMultNode(inputBatch, weights); // BatchSize x OutputDim
        VectorAddNode* weightedBiasedInputs = new VectorAddNode(weightedInputs, bias);
        _nodes.emplace_back(weightedInputs); _nodes.emplace_back(weightedBiasedInputs);

        // apply softmax
        ExpFuncNode* expF = new ExpFuncNode(weightedBiasedInputs); // BatchSize x OutputDim
        ReduceSumNode* expFSum = new ReduceSumNode(expF, 1); // BatchSize x 1
        VectorDivNode* softmax = new VectorDivNode(expF, expFSum); // BatchSize x OutputDim
        _nodes.emplace_back(expF); _nodes.emplace_back(expFSum); _nodes.emplace_back(softmax);

        return softmax;
    }
    NodePtr ConstructLossGraph(NodePtr softmax)
    {
        // add loss function graph part for training
        // compute loss (cross-entropy)
        LogFuncNode* logSoftmax = new LogFuncNode(softmax); // BatchSize x OutputDim
        VectorMultNode* vm = new VectorMultNode(correctClasses, logSoftmax); // BatchSize x OutputDim
        ReduceSumNode* negCrossEntropy = new ReduceSumNode(vm, 1); // BatchSize x 1
        NegateNode* crossEntropy = new NegateNode(negCrossEntropy); // cross entropy per input
        ReduceMeanNode* loss = new ReduceMeanNode(crossEntropy, 0); // 1x1, mean cross entropy over batch
        _nodes.emplace_back(logSoftmax); _nodes.emplace_back(vm);
        _nodes.emplace_back(negCrossEntropy); _nodes.emplace_back(crossEntropy); _nodes.emplace_back(loss);

        return loss;
    }
    void ConstructBackpropagationGraph(NodePtr softmax, NodePtr loss, size_t outputDim)
    {
        // build backpropagation/derivation graph (todo: it would be nice if we could automate this later on)
        NegateNode* negCorrectClasses = new NegateNode(correctClasses); // BatchSize x OutputDim
        VectorAddNode* gradFactors = new VectorAddNode(softmax, negCorrectClasses); // BatchSize x OutputDim
        ReduceMeanNode* biasGrad = new ReduceMeanNode(gradFactors, 0); // 1 x OutputDim
        _nodes.emplace_back(negCorrectClasses); _nodes.emplace_back(gradFactors); _nodes.emplace_back(biasGrad);
        std::vector<NodePtr> weightGradComponents(outputDim);
        for (size_t i = 0; i < outputDim; ++i)
        {
            SliceNode* slice = new SliceNode(gradFactors, i, 1); // slice i from dim 1, BatchSize x 1
            VectorMultNode* tmp = new VectorMultNode(slice, inputBatch); // BatchSize x InputDim
            ReduceMeanNode* reduced = new ReduceMeanNode(tmp, 0); // 1 x InputDim
            _nodes.emplace_back(slice); _nodes.emplace_back(tmp); _nodes.emplace_back(reduced);
            weightGradComponents[i] = reduced;
        }
        StackNode* weightGradTransp = new StackNode(weightGradComponents, 0); // OutputDim x InputDim
        TransposeNode* weightGrad = new TransposeNode(weightGradTransp); // InputDim x OutputDim
        _nodes.emplace_back(weightGradTransp); _nodes.emplace_back(weightGrad);

        const float LearningRate = 0.01f;
        ConstMultNode* biasUpdateDelta = new ConstMultNode(biasGrad, LearningRate); // 1 x OutputDim
        NegateNode* biasUpdateDeltaNeg = new NegateNode(biasUpdateDelta); // 1 x OutputDim
        VectorAddNode* biasUpdated = new VectorAddNode(bias, biasUpdateDeltaNeg); // 1 x OutputDim
        bias->SetInput(biasUpdated);
        _nodes.emplace_back(biasUpdateDelta); _nodes.emplace_back(biasUpdateDeltaNeg); _nodes.emplace_back(biasUpdated);

        ConstMultNode* weightsUpdateDelta = new ConstMultNode(weightGrad, LearningRate); // InputDim x OutputDim
        NegateNode* weightsUpdateDeltaNeg = new NegateNode(weightsUpdateDelta); // InputDim x OutputDim
        VectorAddNode* weightsUpdated = new VectorAddNode(weights, weightsUpdateDeltaNeg); // InputDim x OutputDim
        weights->SetInput(weightsUpdated);
        _nodes.emplace_back(weightsUpdateDelta); _nodes.emplace_back(weightsUpdateDeltaNeg); _nodes.emplace_back(weightsUpdated);
    }
    ConstNodePtr GetWeightsNode() const { return weights; }
    ConstNodePtr GetBiasNode() const { return bias; }
};

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

    // use helper class for graph setup
    MNISTGraph graphTemplate;
    NodePtr softmax = graphTemplate.ConstructClassifierGraph();
    NodePtr loss = graphTemplate.ConstructLossGraph(softmax);
    graphTemplate.ConstructBackpropagationGraph(softmax, loss, OutputDim);

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

    // setup dimension feeding dictionary and compile graph
    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("ImgBatch", MemoryDimensions({BatchSize, InputDim}));
    inputDimensions.emplace("Weights", MemoryDimensions({InputDim, OutputDim}));
    inputDimensions.emplace("Bias", MemoryDimensions({1, OutputDim}));
    inputDimensions.emplace("Classes", MemoryDimensions({BatchSize, OutputDim}));

    GraphCompiler compiler(std::unique_ptr<const ImplementationStrategyFactory>(new ImplementationStrategyFactory));
    const std::unique_ptr<CompiledGraph> graph = compiler.Compile(loss, inputDimensions);

    // initialize graph variables
    InputDataMap variablesDataMap;
    variablesDataMap.emplace("Weights", weightsData);
    variablesDataMap.emplace("Bias", biasData);
    graph->InitializeVariables(variablesDataMap);

    // read input data
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
            imgInputData[i] /= 255.0f; // normalization of input data (with unnormalized data, the exponentiation within the graph might exceed float maximum value)
        }
    }

    // prepare input feeding dict
    InputDataMap inputDataMap;
    inputDataMap.emplace("ImgBatch", imgInputData);
    inputDataMap.emplace("Classes", classesInputData);

    // run training iterations until loss converges
    float previousLoss = std::numeric_limits<float>::infinity();
    float currentLoss = std::numeric_limits<float>::infinity();
    const size_t StopThreshold = 1e-7;
    DataBuffer lossOutput(1);
    do
    {
        previousLoss = currentLoss;
        graph->Evaluate(inputDataMap);

        graph->GetNodeData(loss, lossOutput);
        currentLoss = lossOutput[0];
        std::cout << "Loss: " << currentLoss << std::endl;
    } while (previousLoss - currentLoss > StopThreshold);

    // retrieve trained values for weights and bias
    graph->GetNodeData(graphTemplate.GetWeightsNode(), weightsData);
    graph->GetNodeData(graphTemplate.GetBiasNode(), biasData);

    // set up evaluation graph (forward classification graph only)
    MNISTGraph evaluationGraphTemplate;
    NodePtr evalSoftmax = evaluationGraphTemplate.ConstructClassifierGraph();
    std::unique_ptr<CompiledGraph> evalGraph = compiler.Compile(evalSoftmax, inputDimensions);

    // set trained variable values for evaluation grpah
    evalGraph->InitializeVariables(variablesDataMap);

    // prepare testing/evaluation data
    const size_t TestDataCount = dataset.test_images.size();
    DataBuffer evalInputData(TestDataCount * InputDim);
    {
        DataBuffer::iterator it = std::begin(evalInputData);
        for (size_t i = 0; i < TestDataCount; ++i)
        {
            InputDataBuffer& sampleImageBuffer = dataset.test_images[i];
            it = std::copy(std::begin(sampleImageBuffer), std::end(sampleImageBuffer), it);
        }
        for (size_t i = 0; i < evalInputData.size(); ++i)
        {
            evalInputData[i] /= 255.0f;
        }
    }
    InputDataMap evalInputDataMap;
    evalInputDataMap.emplace("ImgBatch", evalInputData);
    DataBuffer evalPredictions(TestDataCount * OutputDim);

    // run evaluation graph on evaluation input data
    evalGraph->Evaluate(evalInputDataMap);

    // retrieve predictions and calculate classifier precision
    evalGraph->GetNodeData(evalSoftmax, evalPredictions);
    size_t correct = 0;
    for (size_t i = 0; i < TestDataCount; ++i)
    {
        // get index of maximum element in softmax predictions for single sample ( == predicted class )
        DataBuffer::iterator it = std::begin(evalPredictions) + i * OutputDim;
        uint8_t prediction = std::max_element(it, it + OutputDim) - it;

        // is that prediction correct?
        if (prediction == dataset.test_labels[i])
            ++correct;
    }
    float ratio = static_cast<float>(correct) / static_cast<float>(TestDataCount);
    std::cout << correct << " / " << TestDataCount << " (" << ratio << ") classifications are correct." << std::endl;

    return 0;
}
