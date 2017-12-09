#include <iostream>
#include <algorithm>
#include <assert.h>

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
    InputNode* _inputBatch;
    InputNode* _correctClasses;
    VariableNode* _weights;
    VariableNode* _bias;
    float const LearningRate;
public:
    MNISTGraph() : MNISTGraph(0.1f) { }
    MNISTGraph(float const learningRate)
        : _nodes()
        , _inputBatch(new InputNode("ImgBatch")) // constructor sets up input and variable nodes
        , _correctClasses(new InputNode("Classes"))
        , _weights(new VariableNode("Weights"))
        , _bias(new VariableNode("Bias"))
        , LearningRate(learningRate)
    {
        _nodes.emplace_back(_inputBatch), _nodes.emplace_back(_correctClasses);
        _nodes.emplace_back(_weights); _nodes.emplace_back(_bias);
    }
    ~MNISTGraph() { }
    NodePtr ConstructClassifierGraph()
    {
        // build feedforward network for classification
        // weighting inputs and applying bias
        MatrixMultNode* weightedInputs = new MatrixMultNode(_inputBatch, _weights); // BatchSize x OutputDim
        VectorAddNode* weightedBiasedInputs = new VectorAddNode(weightedInputs, _bias);
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
        VectorMultNode* vm = new VectorMultNode(_correctClasses, logSoftmax); // BatchSize x OutputDim
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
        NegateNode* negCorrectClasses = new NegateNode(_correctClasses); // BatchSize x OutputDim
        VectorAddNode* gradFactors = new VectorAddNode(softmax, negCorrectClasses); // BatchSize x OutputDim
        ReduceMeanNode* biasGrad = new ReduceMeanNode(gradFactors, 0); // 1 x OutputDim
        _nodes.emplace_back(negCorrectClasses); _nodes.emplace_back(gradFactors); _nodes.emplace_back(biasGrad);
        std::vector<NodePtr> weightGradComponents(outputDim);
        for (size_t i = 0; i < outputDim; ++i)
        {
            SliceNode* slice = new SliceNode(gradFactors, i, 1); // slice i from dim 1, BatchSize x 1
            VectorMultNode* tmp = new VectorMultNode(slice, _inputBatch); // BatchSize x InputDim
            ReduceMeanNode* reduced = new ReduceMeanNode(tmp, 0); // 1 x InputDim
            _nodes.emplace_back(slice); _nodes.emplace_back(tmp); _nodes.emplace_back(reduced);
            weightGradComponents[i] = reduced;
        }
        StackNode* weightGradTransp = new StackNode(weightGradComponents, 0); // OutputDim x InputDim
        TransposeNode* weightGrad = new TransposeNode(weightGradTransp); // InputDim x OutputDim
        _nodes.emplace_back(weightGradTransp); _nodes.emplace_back(weightGrad);

        ConstMultNode* biasUpdateDelta = new ConstMultNode(biasGrad, LearningRate); // 1 x OutputDim
        NegateNode* biasUpdateDeltaNeg = new NegateNode(biasUpdateDelta); // 1 x OutputDim
        VectorAddNode* biasUpdated = new VectorAddNode(_bias, biasUpdateDeltaNeg); // 1 x OutputDim
        _bias->SetInput(biasUpdated);
        _nodes.emplace_back(biasUpdateDelta); _nodes.emplace_back(biasUpdateDeltaNeg); _nodes.emplace_back(biasUpdated);

        ConstMultNode* weightsUpdateDelta = new ConstMultNode(weightGrad, LearningRate); // InputDim x OutputDim
        NegateNode* weightsUpdateDeltaNeg = new NegateNode(weightsUpdateDelta); // InputDim x OutputDim
        VectorAddNode* weightsUpdated = new VectorAddNode(_weights, weightsUpdateDeltaNeg); // InputDim x OutputDim
        _weights->SetInput(weightsUpdated);
        _nodes.emplace_back(weightsUpdateDelta); _nodes.emplace_back(weightsUpdateDeltaNeg); _nodes.emplace_back(weightsUpdated);
    }
    ConstNodePtr GetWeightsNode() const { return _weights; }
    ConstNodePtr GetBiasNode() const { return _bias; }
};

class MNISTDataset
{
private:
    mnist::MNIST_dataset<std::vector, DataBuffer, uint8_t> _dataset;
    const size_t OutputDim;
    size_t _trainingCursor;
    size_t _testCursor;
public:
    MNISTDataset(const std::string& mnistDataDir, const size_t outputDim)
        : _dataset(mnist::read_dataset_direct<std::vector, DataBuffer, uint8_t>(mnistDataDir, 0, 0))
        , OutputDim(outputDim)
        , _trainingCursor(0)
        , _testCursor(0)
    { }
    size_t GetTrainingSampleCount() const { return _dataset.training_images.size(); }
    size_t GetTestSampleCount() const { return _dataset.test_images.size(); }
    bool GetTrainingBatch(DataBuffer& inputs, DataBuffer& classes, const size_t batchSize)
    {
        std::fill(std::begin(classes), std::end(classes), 0.0f);
        DataBuffer::iterator it = std::begin(inputs);
        for (size_t i = 0; i < batchSize; ++i)
        {
            const size_t id = (_trainingCursor + i) % GetTrainingSampleCount();
            const DataBuffer& sampleImageBuffer = _dataset.training_images[id];
            it = std::copy(std::begin(sampleImageBuffer), std::end(sampleImageBuffer), it);
            uint8_t sampleLabel = _dataset.training_labels[id];
            classes[i * OutputDim + sampleLabel] = 1.0f;
        }
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            inputs[i] /= 255.0f; // normalization of input data (with unnormalized data, the exponentiation within the graph might exceed float maximum value)
        }
        bool wrappedAround = (_trainingCursor + batchSize) >= GetTrainingSampleCount();
        _trainingCursor = (_trainingCursor + batchSize) % GetTrainingSampleCount();
        return wrappedAround;
    }
    bool GetTestBatch(DataBuffer& inputs, std::vector<uint8_t>& labels, const size_t batchSize)
    {
        DataBuffer::iterator it = std::begin(inputs);
        for (size_t i = 0; i < batchSize; ++i)
        {
            const size_t id = (_testCursor + i) % GetTestSampleCount();
            const DataBuffer& sampleImageBuffer = _dataset.test_images[id];
            it = std::copy(std::begin(sampleImageBuffer), std::end(sampleImageBuffer), it);
            labels[i] = _dataset.test_labels[id];
        }
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            inputs[i] /= 255.0f; // normalization of input data (with unnormalized data, the exponentiation within the graph might exceed float maximum value)
        }
        bool wrappedAround = (_testCursor + batchSize) >= GetTestSampleCount();
        _testCursor = (_testCursor + batchSize) % GetTestSampleCount();
        return wrappedAround;
    }
};

int main(int argc, const char * const argv[])
{
    const size_t InputDim = 784;
    const size_t OutputDim = 10;
    size_t BatchSize = 500;
    float StopThreshold = 0.01f;
    float LearningRate = 0.1f;

    // Check command line arguments
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <path to MNIST dataset> [<training batch size> [<learning rate> [<training convergence threshold>]]]" << std::endl;
        return -1;
    }
    const std::string mnistDataDir(argv[1]);

    if (argc > 2)
    {
        BatchSize = std::stoi(argv[2]);
        if (argc > 3)
        {
            LearningRate = std::stof(argv[3]);
            if (argc > 4)
            {
                StopThreshold = std::stof(argv[4]);
            }
        }
    }

    // ####### load training data #######
    MNISTDataset dataset(mnistDataDir, OutputDim);
    assert(dataset.GetTrainingSampleCount() % BatchSize == 0); // to simplify stuff: assert that BatchSize divides available training samples evenly

    // ####### use helper class for graph setup #######
    MNISTGraph graphTemplate(LearningRate);
    NodePtr softmax = graphTemplate.ConstructClassifierGraph();
    NodePtr loss = graphTemplate.ConstructLossGraph(softmax);
    graphTemplate.ConstructBackpropagationGraph(softmax, loss, OutputDim);

    // ####### setup dimension feeding dictionary and compile graph #######
    InputDimensionsMap variableDimensions;
    variableDimensions.emplace("ImgBatch", MemoryDimensions({BatchSize, InputDim}));
    variableDimensions.emplace("Weights", MemoryDimensions({InputDim, OutputDim}));
    variableDimensions.emplace("Bias", MemoryDimensions({1, OutputDim}));
    variableDimensions.emplace("Classes", MemoryDimensions({BatchSize, OutputDim}));

    GraphCompiler compiler(std::unique_ptr<const ImplementationStrategyFactory>(new ImplementationStrategyFactory));
    const std::unique_ptr<CompiledGraph> graph = compiler.Compile(loss, variableDimensions);

    // ####### initialize variables with zero values #######
    DataBuffer weightsData(InputDim * OutputDim);
    DataBuffer biasData(OutputDim);

    std::fill(std::begin(weightsData), std::end(weightsData), 0.0f);
    std::fill(std::begin(biasData), std::end(biasData), 0.0f);

    InputDataMap variablesDataMap;
    variablesDataMap.emplace("Weights", weightsData);
    variablesDataMap.emplace("Bias", biasData);
    graph->InitializeVariables(variablesDataMap);

    // ####### fetch input data #######
    DataBuffer imgInputData(BatchSize * InputDim);
    DataBuffer classesInputData(BatchSize * OutputDim);
    //dataset.GetTrainingBatch(imgInputData, classesInputData, BatchSize);

    // ####### prepare input feeding dict #######
    InputDataMap inputDataMap;
    inputDataMap.emplace("ImgBatch", imgInputData);
    inputDataMap.emplace("Classes", classesInputData);

    // ####### run training iterations until loss converges #######
    float previousLoss = std::numeric_limits<float>::infinity();
    float currentLoss = std::numeric_limits<float>::infinity();
    DataBuffer lossOutput(1);
    do
    {
        previousLoss = currentLoss;
        float epochLoss = 0.0f;
        size_t epochBatchCount = 0;
        bool moreBatches = false;
        do
        {
            moreBatches = dataset.GetTrainingBatch(imgInputData, classesInputData, BatchSize);
            graph->Evaluate(inputDataMap);

            graph->GetNodeData(loss, lossOutput);
            epochLoss += lossOutput[0];
            ++epochBatchCount;
        } while (!moreBatches);

        currentLoss = epochLoss / static_cast<float>(epochBatchCount);
        std::cout << "Loss: " << currentLoss << std::endl;
    } while (previousLoss - currentLoss > StopThreshold);


    // ######################### EVALUATION #########################
    // ####### set up and compile evaluation graph (forward classification graph only) #######
    const size_t TestSampleCount = dataset.GetTestSampleCount();
    MNISTGraph evaluationGraphTemplate;
    NodePtr evalSoftmax = evaluationGraphTemplate.ConstructClassifierGraph();
    variableDimensions.clear();
    variableDimensions.emplace("ImgBatch", MemoryDimensions({TestSampleCount, InputDim}));
    variableDimensions.emplace("Weights", MemoryDimensions({InputDim, OutputDim}));
    variableDimensions.emplace("Bias", MemoryDimensions({1, OutputDim}));
    std::unique_ptr<CompiledGraph> evalGraph = compiler.Compile(evalSoftmax, variableDimensions);

    // ####### initialize variables in evaluation graph with values trained in training graph #######
    graph->GetNodeData(graphTemplate.GetWeightsNode(), weightsData);
    graph->GetNodeData(graphTemplate.GetBiasNode(), biasData);
    evalGraph->InitializeVariables(variablesDataMap);

    // ####### prepare testing/evaluation data #######
    DataBuffer evalInputData(TestSampleCount * InputDim);
    std::vector<uint8_t> evalLabels(TestSampleCount);
    dataset.GetTestBatch(evalInputData, evalLabels, TestSampleCount);

    InputDataMap evalInputDataMap;
    evalInputDataMap.emplace("ImgBatch", evalInputData);
    DataBuffer evalPredictions(TestSampleCount * OutputDim);

    // ####### run evaluation graph on evaluation input data #######
    evalGraph->Evaluate(evalInputDataMap);

    // ####### retrieve predictions and calculate classifier precision #######
    // note: this could also be a graph computation
    evalGraph->GetNodeData(evalSoftmax, evalPredictions);
    size_t correct = 0;
    for (size_t i = 0; i < TestSampleCount; ++i)
    {
        // get index of maximum element in softmax predictions for single sample ( == predicted class )
        DataBuffer::iterator it = std::begin(evalPredictions) + i * OutputDim;
        uint8_t prediction = std::max_element(it, it + OutputDim) - it;

        // is that prediction correct?
        if (prediction == evalLabels[i])
            ++correct;
    }

    // ####### output results #######
    float ratio = static_cast<float>(correct) / static_cast<float>(TestSampleCount);
    std::cout << correct << " / " << TestSampleCount << " (" << ratio << ") classifications are correct." << std::endl;

    return 0;
}
