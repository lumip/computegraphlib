#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
//#include <avxintrin.h>
#include <assert.h>

#include <mnist/mnist_reader.hpp>
#include <papi.h>

// global definitions for data dimensions
const size_t InputDim = 784; // multiple of 8
const size_t BatchSize = 500; //504 or 60000 would be multiples of 8
const size_t OutputDim = 10;

typedef float* DataBuffer;

void initializeData(const std::string& mnistDataDir, DataBuffer& inputs, DataBuffer& classes, DataBuffer& weights, DataBuffer& bias)
{
    // load mnist dataset
    mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset =
            mnist::read_dataset_direct<std::vector, std::vector<float>, uint8_t>(mnistDataDir, 0, 0);

    // copy first batch into input buffer and convert labels into one-hot vector representation
    std::fill_n(classes, BatchSize * OutputDim, 0.0f);
    //DataBuffer::iterator it = std::begin(inputs);
    DataBuffer it = inputs;
    for (size_t i = 0; i < BatchSize; ++i)
    {
        const std::vector<float>& sampleImageBuffer = dataset.training_images[i];
        it = std::copy(std::begin(sampleImageBuffer), std::end(sampleImageBuffer), it);
        uint8_t sampleLabel = dataset.training_labels[i];
        classes[i * OutputDim + sampleLabel] = 1.0f;
    }
    for (size_t i = 0; i < BatchSize * InputDim; ++i)
    {
        inputs[i] /= 255.0f;
        //inputs[i] = i;
    }

    // fill weights and bias with random numbers
    std::mt19937 gen(22); // with fixed seed, reproducible results
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::generate_n(weights, InputDim * OutputDim, [&dist, &gen]()->float {return dist(gen);});
    /*for (size_t i = 0; i < InputDim * OutputDim; ++i)
    {
        weights[i] = i;
    }*/
    std::generate_n(bias, OutputDim, [&dist, &gen]()->float {return dist(gen);}); // we write the bias values into an additional row of inputs
}

inline size_t getIndex(size_t i, size_t j, size_t stride)
{
    return i * stride + j;
}

void multiplyInputAndWeights(const DataBuffer matA, DataBuffer matB, DataBuffer matRes, size_t m, size_t n, size_t d)
{
    assert(d*n % 8 == 0);
    std::fill_n(matRes, m * n, 0.0f);
    for (size_t i = 0; i < m; ++i)
    {
        assert(n == 10); // sorry
        __m256 buf[5];
        for (size_t r = 0; r < 5; ++r)
        {
            buf[r] = _mm256_setzero_ps();
        }
        size_t bufId = 0;
        for (size_t kj = 0; kj < d * n; kj+=8)
        {
            //size_t j = kj % n;
            size_t ks[8] = { (kj+0)/n, (kj+1)/n, (kj+2)/n, (kj+3)/n,
                             (kj+4)/n, (kj+5)/n, (kj+6)/n, (kj+7)/n };
            float aBuf[8]__attribute__ ((aligned (32))) = { matA[getIndex(i, ks[0], d)], matA[getIndex(i, ks[1], d)],
                           matA[getIndex(i, ks[2], d)], matA[getIndex(i, ks[3], d)],
                           matA[getIndex(i, ks[4], d)], matA[getIndex(i, ks[5], d)],
                           matA[getIndex(i, ks[6], d)], matA[getIndex(i, ks[7], d)] };
            __m256 aVec = _mm256_load_ps(aBuf);
            __m256 bVec = _mm256_load_ps(&matB[kj]);
            buf[bufId] = _mm256_add_ps(_mm256_mul_ps(aVec, bVec), buf[bufId]);
            bufId = (bufId + 1) % 5;
        }
        size_t rowStart = getIndex(i, 0, n);
        _mm256_storeu_ps(&matRes[rowStart], buf[0]);
        __m256i mask = _mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0);
        _mm256_maskstore_ps(&matRes[rowStart+8], mask, buf[1]);

        __m256 tmp;
        mask = _mm256_setr_epi32(0, 0, -1, -1, -1, -1, -1, -1);
        tmp  = _mm256_add_ps(_mm256_maskload_ps(&matRes[rowStart-2], mask), buf[1]);
        _mm256_maskstore_ps(&matRes[rowStart-2], mask, tmp);

        mask = _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
        tmp  = _mm256_add_ps(_mm256_maskload_ps(&matRes[rowStart+6], mask), buf[2]);
        _mm256_maskstore_ps(&matRes[rowStart+6], mask, tmp);
        mask = _mm256_setr_epi32(0, 0, 0, 0, -1, -1, -1, -1);
        tmp  = _mm256_add_ps(_mm256_maskload_ps(&matRes[rowStart-4], mask), buf[2]);
        _mm256_maskstore_ps(&matRes[rowStart-4], mask, tmp);

        mask = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);
        tmp  = _mm256_add_ps(_mm256_maskload_ps(&matRes[rowStart+4], mask), buf[3]);
        _mm256_maskstore_ps(&matRes[rowStart+4], mask, tmp);
        mask = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, -1, -1);
        tmp  = _mm256_add_ps(_mm256_maskload_ps(&matRes[rowStart-6], mask), buf[3]);
        _mm256_maskstore_ps(&matRes[rowStart-6], mask, tmp);

        tmp  = _mm256_add_ps(_mm256_loadu_ps(&matRes[rowStart+2]), buf[4]);
        _mm256_storeu_ps(&matRes[rowStart+2], tmp);

        /*for (size_t k = 0; k < d; ++k)
        {
            size_t aIds[8] = { i * n + k }
            __m256 aVec = _mm256_set_ps(matA[getIndex(i, k, d), matA[getINdex]])
            float a_ik = matA[getIndex(i, k, d)]; // linear access to A
            for (size_t j = 0; j < n; ++j)
            {
                matRes[getIndex(i, j, n)] += a_ik * matB[getIndex(k, j, n)]; // linear access to both
            }
        }*/
    }
}

void addBiasAndComputeSoftmax(const DataBuffer& matIn, const DataBuffer& vec, DataBuffer& matRes, size_t m, size_t n)
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

float computeLoss(const DataBuffer& softmax, const DataBuffer& classes, size_t m, size_t n)
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

    DataBuffer inputs = static_cast<DataBuffer>(aligned_alloc(32, (BatchSize * InputDim) * sizeof(float)));
    DataBuffer classes = static_cast<DataBuffer>(aligned_alloc(32, (BatchSize * OutputDim) * sizeof(float)));
    DataBuffer weights = static_cast<DataBuffer>(aligned_alloc(32, (InputDim * OutputDim) * sizeof(float)));
    DataBuffer bias = static_cast<DataBuffer>(aligned_alloc(32, (OutputDim) * sizeof(float)));

    initializeData(mnistDataDir, inputs, classes, weights, bias);

    DataBuffer buffer = static_cast<DataBuffer>(aligned_alloc(32, (BatchSize * OutputDim) * sizeof(float)));

    long long time_start = PAPI_get_real_nsec();
    long long cycs_start = PAPI_get_real_cyc();
    multiplyInputAndWeights(inputs, weights, buffer, BatchSize, OutputDim, InputDim);
    addBiasAndComputeSoftmax(buffer, bias, buffer, BatchSize, OutputDim);
    float loss = computeLoss(buffer, classes, BatchSize, OutputDim);
    long long time_stop = PAPI_get_real_nsec();
    long long cycs_stop = PAPI_get_real_cyc();

    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Computation on " << BatchSize << " samples took " << cycs_stop - cycs_start << " cycles in "<< time_stop - time_start << " ns and " << 0 << " ns to copy input data" << std::endl;

    return 0;
}
