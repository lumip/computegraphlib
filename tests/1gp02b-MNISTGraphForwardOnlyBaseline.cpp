#include <iostream>
#include <algorithm>
#include <cmath>

#include <mnist/mnist_reader.hpp>
#include <papi.h>

// global definitions for data dimensions
const size_t InputDim = 784;
const size_t BatchSize = 500;
const size_t OutputDim = 10;

typedef std::vector<float> DataBuffer;

void initializeData(const std::string& mnistDataDir, DataBuffer& inputs, DataBuffer& classes, DataBuffer& weights, DataBuffer& bias)
{
    // load mnist dataset
    mnist::MNIST_dataset<std::vector, DataBuffer, uint8_t> dataset =
            mnist::read_dataset_direct<std::vector, DataBuffer, uint8_t>(mnistDataDir, 0, 0);

    // copy first batch into input buffer and convert labels into one-hot vector representation
    std::fill(std::begin(classes), std::begin(classes), 0.0f);
    DataBuffer::iterator it = std::begin(inputs);
    for (size_t i = 0; i < BatchSize; ++i)
    {
        const DataBuffer& sampleImageBuffer = dataset.training_images[i];
        it = std::copy(std::begin(sampleImageBuffer), std::end(sampleImageBuffer), it);
        uint8_t sampleLabel = dataset.training_labels[i];
        classes[i * OutputDim + sampleLabel] = 1.0f;
    }
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        inputs[i] /= 255.0f;
    }

    // fill weights and bias with zeroes
    std::fill(std::begin(weights), std::end(weights), 0.0f);
    std::fill(std::begin(bias), std::end(bias), 0.0f);
}

inline size_t getIndex(size_t i, size_t j, size_t stride)
{
    return i * stride + j;
}

inline void multiplyInputAndWeights(const DataBuffer& matA, const DataBuffer& matB, DataBuffer& matRes, size_t m, size_t n, size_t d)
{
    std::fill(std::begin(matRes), std::end(matRes), 0.0f);
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t k = 0; k < d; ++k)
        {
            float a_ik = matA[getIndex(i, k, d)]; // linear access to A
            for (size_t j = 0; j < n; ++j)
            {
                matRes[getIndex(i, j, n)] += a_ik * matB[getIndex(k, j, n)]; // linear access to both
            }
        }
    }
}

inline void addBiasAndComputeSoftmax(const DataBuffer& matIn, const DataBuffer& vec, DataBuffer& matRes, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < n; ++j)
        {
            size_t id = getIndex(i, j, n);
            float expVal = std::exp(matIn[id] + vec[j]);
            sum += expVal;
            matRes[id] = expVal;
        }
        float invSum = 1.0f/sum;
        for (size_t j = 0; j < n; ++j)
        {
            matRes[getIndex(i, j, n)] *= invSum;
        }
    }
}

inline float computeLoss(const DataBuffer& softmax, const DataBuffer& classes, size_t m, size_t n)
{
    float totalSum = 0.0f;
    for (size_t i = 0; i < m; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < n; ++j)
        {
            size_t id = getIndex(i, j, n);
            sum += classes[id] *  std::log(softmax[id]);
        }
        totalSum += (-sum)/static_cast<float>(m);
    }
    return totalSum;
}

int main(int argc, const char * const argv[])
{
    // Check command line arguments
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <path to MNIST dataset>" << std::endl;
        return -1;
    }
    const std::string mnistDataDir(argv[1]);

    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if(retval != PAPI_VER_CURRENT)
    {
        std::cout << "could not initialize PAPI" << std::endl;
        return -1;
    }

    DataBuffer inputs(BatchSize * InputDim);
    DataBuffer classes(BatchSize * OutputDim);
    DataBuffer weights(InputDim * OutputDim);
    DataBuffer bias(OutputDim);

    initializeData(mnistDataDir, inputs, classes, weights, bias);

    DataBuffer buffer(BatchSize * OutputDim);

    long long time_start = PAPI_get_real_nsec();
    multiplyInputAndWeights(inputs, weights, buffer, BatchSize, OutputDim, InputDim);
    addBiasAndComputeSoftmax(buffer, bias, buffer, BatchSize, OutputDim);
    float loss = computeLoss(buffer, classes, BatchSize, OutputDim);
    long long time_stop = PAPI_get_real_nsec();

    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Setup: 0 ns; Copy: 0 ns; Compute: " << time_stop - time_start << " ns" << std::endl;

    return 0;
}
