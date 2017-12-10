#include "MatrixMultNodeGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"
#include "../OCLWrappers.hpp"

// performance with 0sc03g
// b global  / a global : 4041644587 ns - 4.042 s
// b private / a global : 3661179855 ns - 3.661 s
// b global  / a local  : 2921280578 ns - 2.942 s
// b private / a local  : 2665492501 ns - 2.665 s
// might all be affected by data copying times (not sure these are really not factored in the measurements as of yet)
const std::string MatrixMultNodeGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float* matA, __global float* matB, __global float* matResult, uint m, uint n, uint d, __local float* a_i)
{
    uint j = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsiz = get_local_size(0);

    float b_j[10000];
    for (uint k = 0; k < d; ++k)
    {
        b_j[k] = matB[k * n + j];
    }

    for (uint i = 0; i < m; ++i)
    {
        for (uint k = lid; k < d; k += lsiz)
        {
            a_i[k] = matA[i * d + k];
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        float val = 0.0f;
        for (uint k = 0; k < d; ++k)
        {
            //val += matA[i * d + k] * matB[k * n + j];
            //val += a_i[k] * matB[k * n + j];
            //val += matA[i * d + k] * b_j[k];
            val += a_i[k] * b_j[k];
        }
        matResult[i * n + j] = val;
    }
}
)==kernel==";

// computing rows per work items is less efficient than computing columns per work item (probable cause: memory access to not coalesce as nicely)
// performance with 0sc03g
// a global  / b global : 11218764530 ns - 11.219 s
// a private / b global : 4865126016 ns - 4.865 s
// a global  / b local  : 13960163008 ns - 13.96 s
// a private / b local  : 2782160007 ns - 2.782 s
// might all be affected by data copying times (not sure these are really not factored in the measurements as of yet)
/*const std::string MatrixMultNodeGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float* matA, __global float* matB, __global float* matResult, uint m, uint n, uint d, __local float* b_j)
{
    uint i = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsiz = get_local_size(0);

    float a_i[10000];
    for (uint k = 0; k < d; ++k)
    {
        a_i[k] = matA[i * d + k];
    }

    for (uint j = 0; j < n; ++j)
    {
        for (uint k = lid; k < d; k += lsiz)
        {
            b_j[k] = matB[k * n + j];
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        float val = 0.0f;
        for (uint k = 0; k < d; ++k)
        {
            val += a_i[k] * b_j[k];
            //val += a_i[k] * matB[k * n + j];
            //val += matA[i * d + k] * b_j[k];
            //val += matA[i * d + k] * matB[k * n + j];
        }
        matResult[i * n + j] = val;
    }
}
)==kernel==";*/

/*
const std::string MatrixMultNodeGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float* matA, __global float* matB, __global float* matResult, uint n, uint d)
{
    uint i = get_global_id(0);

    for (uint j = 0; j < n; ++j)
    {
        float val = 0.0f;
        for (uint k = 0; k < d; ++k)
        {
            val += matA[i * d + k] * matB[k * n + j];
        }
        matResult[i * n + j] = val;
    }
}
)==kernel==";*/

/*MatrixMultNodeGPUKernel::MatrixMultNodeGPUKernel(OpenCLCompiler& compiler, const cl_command_queue queue, cl_mem memA, cl_mem memB, cl_mem memRes, size_t m, size_t n, size_t d)
    : _kernel(compiler.CompileKernel(KernelSource)), _queue(queue), _memA(memA), _memB(memB), _memRes(memRes), _m(m), _n(n), _d(d)*/
MatrixMultNodeGPUKernel::MatrixMultNodeGPUKernel(OpenCLCompiler& compiler, const cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memA, cl_mem memB, cl_mem memRes, size_t m, size_t n, size_t d)
    : GPUKernel(queue, compiler.CompileKernel(KernelSource), inputKernels), _memA(memA), _memB(memB), _memRes(memRes), _m(m), _n(n), _d(d)
{ }

MatrixMultNodeGPUKernel::~MatrixMultNodeGPUKernel() { }

void MatrixMultNodeGPUKernel::Run()
{
    clSetKernelArg(_kernel, 0, sizeof(cl_mem), &_memA);
    clSetKernelArg(_kernel, 1, sizeof(cl_mem), &_memB);
    clSetKernelArg(_kernel, 2, sizeof(cl_mem), &_memRes);
    clSetKernelArg(_kernel, 3, sizeof(cl_uint), &_m);
    clSetKernelArg(_kernel, 4, sizeof(cl_uint), &_n);
    clSetKernelArg(_kernel, 5, sizeof(cl_uint), &_d);
    clSetKernelArg(_kernel, 6, sizeof(float) * _d, nullptr);
    size_t globalWorkSize[1] = { _n };
    std::vector<cl_event> inputEvents = GetNodeInputEvents();
    cl_event ownEvent;
    OCLWrappers::CheckCLError(
        clEnqueueNDRangeKernel(_queue, _kernel, 1, nullptr, globalWorkSize, nullptr, inputEvents.size(), inputEvents.data(), &ownEvent)
    , "clEnqueueNDRangeKernel (for MatrixMultNodeGPUKernel)");
    SetEvent(ownEvent);
}
