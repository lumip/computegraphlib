#include <iostream>
#include <algorithm>
#include <cmath>
#include "immintrin.h"
#include <assert.h>

#include <mnist/mnist_reader.hpp>
#include <papi.h>

// global definitions for data dimensions
const size_t InputDim = 784;
const size_t InputDimMem = 784; // multiple of 8
size_t BatchSize = 500;
size_t BatchSizeMem = 504; // multiple of 8
const size_t OutputDim = 10;
const size_t OutputDimMem = 16; // multiple of 8

typedef float* DataBuffer;

inline size_t getIndex(size_t i, size_t j, size_t stride)
{
    return i * stride + j;
}

void initializeData(const std::string& mnistDataDir, DataBuffer& inputs, DataBuffer& classes, DataBuffer& weights, DataBuffer& bias)
{
    // load mnist dataset
    mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset =
            mnist::read_dataset_direct<std::vector, std::vector<float>, uint8_t>(mnistDataDir, 0, 0);

    // copy first batch into input buffer and convert labels into one-hot vector representation
    std::fill_n(classes, BatchSizeMem * OutputDimMem, 0.0f);
    for (size_t i = 0; i < BatchSize; ++i)
    {
        const std::vector<float>& sampleImageBuffer = dataset.training_images[i];
        std::copy(std::begin(sampleImageBuffer), std::end(sampleImageBuffer), inputs + i * InputDimMem);
        uint8_t sampleLabel = dataset.training_labels[i];
        classes[i * OutputDimMem + sampleLabel] = 1.0f;
    }
    for (size_t i = 0; i < BatchSizeMem * InputDimMem; ++i)
    {
        inputs[i] /= 255.0f;
    }

    // fill weights and bias with zeroes
    std::fill_n(weights, InputDimMem * OutputDimMem, 0.0f);
    std::fill_n(bias, OutputDimMem, 0.0f);
}

inline void multiplyInputAndWeights(const DataBuffer matA, DataBuffer matB, DataBuffer matRes)
{
    assert(OutputDimMem % 8 == 0); // :(
    std::fill_n(matRes, BatchSizeMem * OutputDimMem, 0.0f);
    for (size_t i = 0; i < BatchSize; ++i)
    {
        for (size_t k = 0; k < InputDim; ++k)
        {
            __m256 aVec = _mm256_set1_ps(matA[getIndex(i, k, InputDimMem)]);
            for (size_t j = 0; j < OutputDim; j+=8)
            {
                __m256 bVec = _mm256_load_ps(&matB[getIndex(k, j, OutputDimMem)]);
                __m256 rVec = _mm256_load_ps(&matRes[getIndex(i, j, OutputDimMem)]);
                rVec = _mm256_add_ps(_mm256_mul_ps(aVec, bVec), rVec);
                _mm256_store_ps(&matRes[getIndex(i, j, OutputDimMem)], rVec);
            }
        }
    }
}

inline __m256 sumHorizontally(__m256 xVec)
{
    __m256 zeroVec = _mm256_setzero_ps();
    xVec = _mm256_hadd_ps(_mm256_hadd_ps(xVec, zeroVec), zeroVec); // sum adjacent entries in each 128b half
    xVec = _mm256_add_ps(_mm256_permute2f128_ps(xVec, xVec, 1), xVec); // swap 128b halves and add that, giving complete sum in pos 0 and 4
    xVec = _mm256_permute_ps(xVec, 0b00000000); // distribute sums over entire vector
    return xVec;
}

inline void addBiasAndComputeSoftmax(const DataBuffer& matIn, const DataBuffer& vec, DataBuffer& matRes)
{
    assert(OutputDimMem % 8 == 0);
    for (size_t i = 0; i < BatchSize; ++i)
    {
        for (size_t j = 0; j < OutputDim; j+=8)
        {
            size_t id = getIndex(i, j, OutputDimMem);
            __m256 mVec = _mm256_load_ps(&matIn[id]);
            __m256 vVec = _mm256_load_ps(&vec[j]);
            __m256 rVec = _mm256_add_ps(mVec, vVec);
            _mm256_store_ps(&matRes[id], rVec);
        }
        for (size_t j = 0; j < OutputDim; ++j) // can't SIMDize this?
        {
            size_t id = getIndex(i, j, OutputDimMem);
            matRes[id] = std::exp(matRes[id]);
        }
        __m256 sumVec = _mm256_setzero_ps();
        for (size_t j = 0; j < OutputDim; j+=8)
        {
            size_t id = getIndex(i, j, OutputDimMem);
            __m256 aVec = _mm256_load_ps(&matRes[id]);
            sumVec = _mm256_add_ps(aVec, sumVec);
        }
        sumVec = sumHorizontally(sumVec);
        __m256 invSumVec = _mm256_rcp_ps(sumVec);
        for (size_t j = 0; j < OutputDim; j+=8)
        {
            size_t id = getIndex(i, j, OutputDimMem);
            __m256 rVec = _mm256_load_ps(&matRes[id]);
            rVec = _mm256_mul_ps(invSumVec, rVec);
            _mm256_store_ps(&matRes[id], rVec);
        }
    }
}

inline float computeLoss(DataBuffer& softmax, const DataBuffer& classes)
{
    assert(OutputDimMem % 8 == 0);
    __m256 invBatchSizeVec = _mm256_set1_ps(1/static_cast<float>(BatchSize));
    __m256 totalSumVec = _mm256_setzero_ps();
    for (size_t i = 0; i < BatchSize; ++i)
    {
        for (size_t j = 0; j < OutputDim; ++j)  // can't SIMDize this?
        {
            size_t id = getIndex(i, j, OutputDimMem);
            softmax[id] = std::log(softmax[id]);
        }

        __m256 sumVec = _mm256_setzero_ps();
        for (size_t j = 0; j < OutputDim; j+=8)
        {
            size_t id = getIndex(i, j, OutputDimMem);
            __m256 cVec = _mm256_load_ps(&classes[id]);
            __m256 sVec = _mm256_load_ps(&softmax[id]);
            sumVec = _mm256_add_ps(_mm256_mul_ps(cVec, sVec), sumVec);
        }
        totalSumVec = _mm256_sub_ps(totalSumVec, _mm256_mul_ps(sumVec, invBatchSizeVec));
    }
    totalSumVec = sumHorizontally(totalSumVec);
    float totalSum[8] __attribute__((aligned(32))) { 0.0f };
    _mm256_store_ps(totalSum, totalSumVec);
    return totalSum[0];
}

int main(int argc, const char * const argv[])
{
    // Check command line arguments
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <path to MNIST dataset> [<training batch size>]" << std::endl;
        return -1;
    }
    const std::string mnistDataDir(argv[1]);

    if (argc > 2)
    {
        BatchSize = std::stoi(argv[2]);
        BatchSizeMem = ((BatchSize / 8) + 1) * 8; // make BatchSizeMem the next larger multiple of 8 of BatchSize
    }

    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if(retval != PAPI_VER_CURRENT)
    {
        std::cout << "could not initialize PAPI" << std::endl;
        return -1;
    }

    DataBuffer inputs = static_cast<DataBuffer>(aligned_alloc(32, (BatchSizeMem * InputDimMem) * sizeof(float)));
    DataBuffer classes = static_cast<DataBuffer>(aligned_alloc(32, (BatchSizeMem * OutputDimMem) * sizeof(float)));
    DataBuffer weights = static_cast<DataBuffer>(aligned_alloc(32, (InputDimMem * OutputDimMem) * sizeof(float)));
    DataBuffer bias = static_cast<DataBuffer>(aligned_alloc(32, (OutputDimMem) * sizeof(float)));

    initializeData(mnistDataDir, inputs, classes, weights, bias);

    DataBuffer buffer = static_cast<DataBuffer>(aligned_alloc(32, (BatchSizeMem * OutputDimMem) * sizeof(float)));

    long long time_start = PAPI_get_real_nsec();;
    multiplyInputAndWeights(inputs, weights, buffer);
    addBiasAndComputeSoftmax(buffer, bias, buffer);
    float loss = computeLoss(buffer, classes);
    long long time_stop = PAPI_get_real_nsec();

    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Setup: 0 ns; Copy: 0 ns; Compute: " << time_stop - time_start << " ns" << std::endl;

    return 0;
}
