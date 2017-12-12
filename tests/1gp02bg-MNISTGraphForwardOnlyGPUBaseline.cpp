#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>

#include <CL/cl.h>
#include <mnist/mnist_reader.hpp>
#include <papi.h>

/*
 *
 *
 * THIS WAS INTENDED TO SERVE AS A BASELINE FOR HOW FAST AN OPENCL IMPLEMENTATION CAN GET
 * TO COMPARE THE PERFORMANCE OF THE MORE ABSTRACTED FRAMEWORK
 *  - it should be faster since it can make assumptions about the problem
 *    and thus perform the same work in fewer kernels and buffers ( = less overhead)
 *  - in particular:
 *    - all matrices/vectors were transposed which, in principle, allows the functions
 *      summing over all elements of the output dimension to coalesce memory accesses better
 *    - only three kernels are used instead of one for every operation:
 *      - first kernel performs regular matrix multiplication (Wt * Xt)
 *      - second kernel add bias (per column), and computes softmax value per element
 *              (exponentiates, sums over column and divides all entries)
 *      - third kernel computes partial loss per elements
 *          (takes log of softmax, multiplies with correct label, takes mean over column)
 *      - fourth kernel sums up partial losses
 *
 * IT TURNED OUT, HOWEVER, TO BE ACTUALLY SLOWER THAN THE FRAMEWORK IMPLEMENTATION
 * IT ALSO DOES NOT COMPUTE THE CORRECT RESULT
 *  - and I don't have time to figure out why now
 *
 * THUS: ABANDONED FOR NOW
 *
 */

// global definitions for data dimensions
const size_t InputDim = 784;
const size_t BatchSize = 500;
const size_t OutputDim = 10;

cl_context clContext;
cl_command_queue clQueue;
cl_device_id clDevice;

extern std::string const multiplyInputAndWeightsKernelSource;
extern std::string const addBiasAndComputeSoftmaxKernelSource;
extern std::string const computeLossKernelSource;

typedef std::vector<float> DataBuffer;

void CheckCLError(cl_int status, std::string const& method)
{
    if (status != CL_SUCCESS)
    {
        throw std::runtime_error("OpenCL error while calling " + method);
    }
}

void setUpOpenCL()
{
    cl_platform_id platformId;
    CheckCLError(clGetPlatformIDs(1, &platformId, nullptr), "clGetPlatformIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformId, 0
    };
    cl_int status = CL_SUCCESS;
    clContext = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &status);
    CheckCLError(status, "clCreateContextFromType");

    size_t contextDeviceCount = 0;
    CheckCLError(clGetContextInfo(clContext, CL_CONTEXT_DEVICES, 0, nullptr, &contextDeviceCount), "clGetContextInfo");
    if (contextDeviceCount == 0)
    {
        throw std::runtime_error("Given context does not contain devices.");
    }
    std::vector<cl_device_id> devices(contextDeviceCount);
    CheckCLError(clGetContextInfo(clContext, CL_CONTEXT_DEVICES, contextDeviceCount, devices.data(), nullptr), "clGetContextInfo");
    clDevice = devices[0];
    cl_queue_properties queueProperties[] =
    {
        CL_QUEUE_PROPERTIES,
        (cl_queue_properties)CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0
    };
    clQueue = clCreateCommandQueueWithProperties(clContext, clDevice, queueProperties, &status);
    CheckCLError(status, "clCreateCommandQueue");
}

cl_program CompileProgram()
{
    const size_t lengths[3] = { multiplyInputAndWeightsKernelSource.size(),
                                addBiasAndComputeSoftmaxKernelSource.size(),
                                computeLossKernelSource.size() };
    const char* sources[3] { multiplyInputAndWeightsKernelSource.c_str(),
                                   addBiasAndComputeSoftmaxKernelSource.c_str(),
                                   computeLossKernelSource.c_str() };
    cl_int status = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(clContext, 3, sources, lengths, &status);
    CheckCLError(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (status == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t buildLogSize = 0;
        CheckCLError(clGetProgramBuildInfo(program, clDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildLogSize), "clGetProgramBuildInfo");
        std::vector<char> buildLog(buildLogSize);
        CheckCLError(clGetProgramBuildInfo(program, clDevice, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog.data(), nullptr), "clGetProgramBuildInfo");
        std::stringstream ss;
        ss << "clBuildProgram: GPU kernel did not compile! Error Message:\n" << buildLog.data();
        throw std::runtime_error(ss.str());
    }
    else
    {
        CheckCLError(status, "clBuildProgram");
    }
    return program;
}

void initializeData(const std::string& mnistDataDir, float* const inputs, float* const classes, float* const weights, float* const bias)
{
    // load mnist dataset
    mnist::MNIST_dataset<std::vector, DataBuffer, uint8_t> dataset =
            mnist::read_dataset_direct<std::vector, DataBuffer, uint8_t>(mnistDataDir, 0, 0);

    // copy first batch into input buffer and convert labels into one-hot vector representation
    std::fill_n(classes, BatchSize * OutputDim, 0.0f);
    float* it = inputs;
    for (size_t i = 0; i < BatchSize; ++i)
    {
        const DataBuffer& sampleImageBuffer = dataset.training_images[i];
        for (size_t j = 0; j < InputDim; ++j)
        {
            inputs[j * BatchSize + i] = sampleImageBuffer[j];
        }
        it = std::copy(std::begin(sampleImageBuffer), std::end(sampleImageBuffer), it);
        uint8_t sampleLabel = dataset.training_labels[i];
        classes[sampleLabel * BatchSize + i] = 1.0f;
    }
    for (size_t i = 0; i < BatchSize * InputDim; ++i)
    {
        inputs[i] /= 255.0f;
    }

    // fill weights and bias with zeroes
    std::fill_n(weights, InputDim * OutputDim, 0.0f);
    std::fill_n(bias, 1 * OutputDim, 0.0f);
}

std::string const multiplyInputAndWeightsKernelSource =
R"==kernel==(
    __kernel void multiplyInputAndWeights(__global float const * const matA, __global float const * const matB, __global float* const matResult, uint const m, uint const n, uint const d, __local float* const a_i)
    {
        uint j = get_global_id(0);
        uint lid = get_local_id(0);
        uint lsiz = get_local_size(0);

        for (uint i = 0; i < m; ++i)
        {
            for (uint k = lid; k < d; k += lsiz)
            {
                a_i[k] = matA[i * d + k];
            }
            work_group_barrier(CLK_LOCAL_MEM_FENCE);

            if (j < n) // if worker is outside of working set, do not compute. (IMPORTANT: the worker must! be active in the loop above)
            {
                float val = 0.0f;
                for (uint k = 0; k < d; ++k)
                {
                    val += a_i[k] * matB[k * n + j];
                }
                matResult[i * n + j] = val;
            }
        }
    }
)==kernel==";

// assuming we compute on transposed matrices for greater efficiency
std::string const addBiasAndComputeSoftmaxKernelSource =
R"==kernel==(
    #define n 10 // OutputDim, here: nr of rows
    __kernel void addBiasAndComputeSoftmax(__global float const * const matIn, __global float const * const vec, __global float* const matRes, uint const m)
    {
        uint i = get_global_id(0); // i is a row of the untransposed matrix, so a column in here
        uint lid = get_local_id(0);
        uint lsiz = get_local_size(0);
        if (i >= m) return;

        __local float l_vec[n];
        for (uint k = lid; k < n; k += lsiz)
        {
            l_vec[k] = vec[k];
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
        float sum = 0.0f;
        for (uint j = 0; j < n; ++j)
        {
            uint id = j * n + i;
            float expVal = exp(matIn[id] + l_vec[j]);
            sum += expVal;
            matRes[id] = expVal;
        }
        float invSum = 1.0/sum;
        for (uint j = 0; j < n; ++j)
        {
            uint id = id;
            matRes[id] *= invSum;
        }
    }
)==kernel==";

std::string const computeLossKernelSource =
R"==kernel==(
    #define n 10 // OutputDim, here: nr of rows
    __kernel void computeLossPerElement(__global float* const softmax, __global float const * const classes, uint const m)
    {
        uint i = get_global_id(0); // i is a row of the untransposed matrix, so a column in here
        if (i >= m) return;

        float sum = 0.0f;
        for (size_t j = 0; j < n; ++j)
        {
            uint id = j * n + i;
            sum += classes[id] * log(softmax[id]);
        }
        softmax[i] = -sum;
    }
    /*__kernel void sumTotalLoss(__global float* const lossPerElement, uint const m) // here we will assume to have 32 workers just summing up the first row together
    {
        uint id = get_global_id(0); // i is a row of the untransposed matrix, so a column in here
        uint lid = get_local_id(0);
        uint lsiz = get_local_size(0);
        __local float l_vec[32]; // size of workgroup

        float sum = 0.0f;
        for (uint i = lid; i < m; i += lsiz)
        {
            sum += lossPerElement[i];
        }
        l_vec[id] = sum;
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
        if (id < 16)
        {
            l_vec[id] += l_vec[id+16];
            work_group_barrier(CLK_LOCAL_MEM_FENCE);
            if (id < 8)
            {
                l_vec[id] += l_vec[id+8];
                work_group_barrier(CLK_LOCAL_MEM_FENCE);
                if (id < 4)
                {
                    l_vec[id] += l_vec[id+4];
                    work_group_barrier(CLK_LOCAL_MEM_FENCE);
                    if (id < 2)
                    {
                        l_vec[id] += l_vec[id+2];
                        work_group_barrier(CLK_LOCAL_MEM_FENCE);
                        if (id == 0)
                        {
                            lossPerElement[0] = (l_vec[0] + l_vec[1])/(float)m;
                        }
                    }
                }
            }
        }
    }*/
    __kernel void sumTotalLoss(__global float* const lossPerElement, uint const m) // here we will assume to have 32 workers just summing up the first row together
    {
        uint id = get_global_id(0);
        if (id != 0) return;
        float sum = 0.0f;
        for (uint i = 0; i < m; ++i)
        {
            sum += lossPerElement[i];
        }
        lossPerElement[0] = sum / (float)m;
    }
)==kernel==";

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

    setUpOpenCL();

    long long time_setup_start = PAPI_get_real_nsec();

    cl_int status = CL_SUCCESS;
    cl_mem memInputsDev = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, BatchSize * InputDim * sizeof(float), nullptr, &status);
    CheckCLError(status, "clCreateBuffer for memInputsDev");
    cl_mem memClassesDev = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, BatchSize * OutputDim * sizeof(float), nullptr, &status);
    CheckCLError(status, "clCreateBuffer for memClassesDev");
    cl_mem memWeightsDev = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, InputDim * OutputDim * sizeof(float), nullptr, &status);
    CheckCLError(status, "clCreateBuffer for memWeightsDev");
    cl_mem memBiasDev = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 1 * OutputDim * sizeof(float), nullptr, &status);
    CheckCLError(status, "clCreateBuffer for memBiasDev");
    cl_mem memBuffer = clCreateBuffer(clContext, CL_MEM_READ_WRITE, BatchSize * OutputDim * sizeof(float), nullptr, &status);
    CheckCLError(status, "clCreateBuffer for memBuffer");

    cl_program program = CompileProgram();
    cl_kernel multiplyInputAndWeightsKernel = clCreateKernel(program, "multiplyInputAndWeights", &status);
    CheckCLError(status, "clCreateKernel for multiplyInputAndWeightsKernel");
    cl_kernel addBiasAndComputeSoftmaxKernel = clCreateKernel(program, "addBiasAndComputeSoftmax", &status);
    CheckCLError(status, "clCreateKernel for addBiasAndComputeSoftmaxKernel");
    cl_kernel computeLossPerElementKernel = clCreateKernel(program, "computeLossPerElement", &status);
    CheckCLError(status, "clCreateKernel for computeLossPerElementKernel");
    cl_kernel sumTotalLossKernel = clCreateKernel(program, "sumTotalLoss", &status);
    CheckCLError(status, "clCreateKernel for sumTotalLossKernel");

    float* inputsBuffer = static_cast<float*>(clEnqueueMapBuffer(clQueue, memInputsDev, CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, 0, BatchSize * InputDim * sizeof(float), 0, nullptr, nullptr, &status));
    CheckCLError(status, "clEnqueueMapBuffer for memInpusDev");
    float* classesBuffer = static_cast<float*>(clEnqueueMapBuffer(clQueue, memClassesDev, CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, 0, BatchSize * OutputDim * sizeof(float), 0, nullptr, nullptr, &status));
    CheckCLError(status, "clEnqueueMapBuffer for memClassesDev");
    float* weightsBuffer = static_cast<float*>(clEnqueueMapBuffer(clQueue, memWeightsDev, CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, 0, InputDim * OutputDim * sizeof(float), 0, nullptr, nullptr, &status));
    CheckCLError(status, "clEnqueueMapBuffer for memWeightsDev");
    float* biasBuffer = static_cast<float*>(clEnqueueMapBuffer(clQueue, memBiasDev, CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, 0, 1 * OutputDim * sizeof(float), 0, nullptr, nullptr, &status));
    CheckCLError(status, "clEnqueueMapBuffer for memBiasDev");
    clFinish(clQueue);

    long long time_setup_stop = PAPI_get_real_nsec();

    initializeData(mnistDataDir, inputsBuffer, classesBuffer, weightsBuffer, biasBuffer);

    long long time_start = PAPI_get_real_nsec();

    cl_event inputsEvent;
    cl_event classesEvent;
    cl_event weightsEvent;
    cl_event biasEvent;
    CheckCLError(clEnqueueUnmapMemObject(clQueue, memInputsDev, inputsBuffer, 0, nullptr, &inputsEvent), "clEnqueueUnmapEvent for memInputsDev");
    CheckCLError(clEnqueueUnmapMemObject(clQueue, memWeightsDev, weightsBuffer, 0, nullptr, &weightsEvent), "clEnqueueUnmapEvent for memWeightsDev");
    CheckCLError(clEnqueueUnmapMemObject(clQueue, memBiasDev, biasBuffer, 0, nullptr, &biasEvent), "clEnqueueUnmapEvent for memBiasDev");
    CheckCLError(clEnqueueUnmapMemObject(clQueue, memClassesDev, classesBuffer, 0, nullptr, &classesEvent), "clEnqueueUnmapEvent for memClassesDev");

    cl_event multiplyInputAndWeightsEvent;
    clSetKernelArg(multiplyInputAndWeightsKernel, 0, sizeof(cl_mem), &memWeightsDev);
    clSetKernelArg(multiplyInputAndWeightsKernel, 1, sizeof(cl_mem), &memInputsDev);
    clSetKernelArg(multiplyInputAndWeightsKernel, 2, sizeof(cl_mem), &memBuffer);
    clSetKernelArg(multiplyInputAndWeightsKernel, 3, sizeof(cl_uint), &OutputDim);
    clSetKernelArg(multiplyInputAndWeightsKernel, 4, sizeof(cl_uint), &BatchSize);
    clSetKernelArg(multiplyInputAndWeightsKernel, 5, sizeof(cl_uint), &InputDim);
    clSetKernelArg(multiplyInputAndWeightsKernel, 6, sizeof(cl_mem) * InputDim, nullptr);
    size_t globalWorkSize[1] = { 512 };
    size_t localWorkSize[1] = { 256 };
    cl_event const multiplyInputAndWeightsWaitList[2] = { inputsEvent, weightsEvent };
    CheckCLError(clEnqueueNDRangeKernel(clQueue, multiplyInputAndWeightsKernel, 1, nullptr, globalWorkSize, localWorkSize, 2, multiplyInputAndWeightsWaitList, &multiplyInputAndWeightsEvent), "clEnqeueNDRangeKernel for multiplyInputAndWeightsKernel");

    cl_event addBiasAndComputeSoftmaxEvent;
    clSetKernelArg(addBiasAndComputeSoftmaxKernel, 0, sizeof(cl_mem), &memBuffer);
    clSetKernelArg(addBiasAndComputeSoftmaxKernel, 1, sizeof(cl_mem), &memBiasDev);
    clSetKernelArg(addBiasAndComputeSoftmaxKernel, 2, sizeof(cl_mem), &memBuffer);
    clSetKernelArg(addBiasAndComputeSoftmaxKernel, 3, sizeof(cl_uint), &BatchSize);
    cl_event const addBiasAndComputeSoftmaxWaitList[2] = { biasEvent, multiplyInputAndWeightsEvent };
    CheckCLError(clEnqueueNDRangeKernel(clQueue, addBiasAndComputeSoftmaxKernel, 1, nullptr, globalWorkSize, localWorkSize, 2, addBiasAndComputeSoftmaxWaitList, &addBiasAndComputeSoftmaxEvent), "clEnqeueNDRangeKernel for addBiasAndComputeSoftmaxKernel");

    cl_event computeLossPerElementEvent;
    clSetKernelArg(computeLossPerElementKernel, 0, sizeof(cl_mem), &memBuffer);
    clSetKernelArg(computeLossPerElementKernel, 1, sizeof(cl_mem), &memClassesDev);
    clSetKernelArg(computeLossPerElementKernel, 2, sizeof(cl_uint), &BatchSize);
    cl_event const computeLossPerElementWaitList[2] = { classesEvent, addBiasAndComputeSoftmaxEvent };
    CheckCLError(clEnqueueNDRangeKernel(clQueue, computeLossPerElementKernel, 1, nullptr, globalWorkSize, localWorkSize, 2, computeLossPerElementWaitList, &computeLossPerElementEvent), "clEnqeueNDRangeKernel for computeLossPerElementKernel");

    cl_event sumTotalLossEvent;
    {
        clSetKernelArg(sumTotalLossKernel, 0, sizeof(cl_mem), &memBuffer);
        clSetKernelArg(sumTotalLossKernel, 1, sizeof(cl_uint), &BatchSize);
        size_t const globalWorkSize[1] = { 1 };
        size_t const localWorkSize[1] = { 1 };
        CheckCLError(clEnqueueNDRangeKernel(clQueue, sumTotalLossKernel, 1, nullptr, globalWorkSize, localWorkSize, 1, &computeLossPerElementEvent, &sumTotalLossEvent), "clEnqeueNDRangeKernel for sumTotalLossKernel");
    }
    float loss;
    clEnqueueReadBuffer(clQueue, memBuffer, CL_TRUE, 0, sizeof(float), &loss, 1, &computeLossPerElementEvent, nullptr);
    long long time_stop = PAPI_get_real_nsec();

    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Setup: " << time_setup_stop - time_setup_start << " ns; Copy: -1 ns; Compute+Copy: " << time_stop - time_start << " ns" << std::endl;

    clReleaseEvent(sumTotalLossEvent);
    clReleaseEvent(computeLossPerElementEvent);
    clReleaseEvent(addBiasAndComputeSoftmaxEvent);
    clReleaseEvent(multiplyInputAndWeightsEvent);
    clReleaseEvent(biasEvent);
    clReleaseEvent(weightsEvent);
    clReleaseEvent(classesEvent);
    clReleaseEvent(inputsEvent);

    clReleaseKernel(sumTotalLossKernel);
    clReleaseKernel(computeLossPerElementKernel);
    clReleaseKernel(addBiasAndComputeSoftmaxKernel);
    clReleaseKernel(multiplyInputAndWeightsKernel);
    clReleaseProgram(program);

    clReleaseMemObject(memBuffer);
    clReleaseMemObject(memBiasDev);
    clReleaseMemObject(memWeightsDev);
    clReleaseMemObject(memClassesDev);
    clReleaseMemObject(memInputsDev);

    clReleaseCommandQueue(clQueue);
    clReleaseContext(clContext);

    return 0;
}
