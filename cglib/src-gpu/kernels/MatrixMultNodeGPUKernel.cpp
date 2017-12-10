#include "MatrixMultNodeGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"
#include "../OCLWrappers.hpp"

const std::string MatrixMultNodeGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float* matA, __global float* matB, __global float* matResult, uint dim)
{
    uint i = get_global_id(0);
    uint j = get_global_id(1);
    uint m = get_global_size(0);
    uint n = get_global_size(1);
    float val = 0.0f;
    for (uint k = 0; k < dim; ++k)
    {
        val += matA[i * dim + k] * matB[k * n + j];
    }
    matResult[i * n + j] = val;
}
)==kernel==";

MatrixMultNodeGPUKernel::MatrixMultNodeGPUKernel(OpenCLCompiler& compiler, const cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memA, cl_mem memB, cl_mem memRes, size_t m, size_t n, size_t d)
    : GPUKernel(queue, compiler.CompileKernel(KernelSource), inputKernels), _memA(memA), _memB(memB), _memRes(memRes), _m(m), _n(n), _d(d)
{ }

MatrixMultNodeGPUKernel::~MatrixMultNodeGPUKernel() { }

void MatrixMultNodeGPUKernel::Run()
{
    clSetKernelArg(_kernel, 0, sizeof(cl_mem), &_memA);
    clSetKernelArg(_kernel, 1, sizeof(cl_mem), &_memB);
    clSetKernelArg(_kernel, 2, sizeof(cl_mem), &_memRes);
    clSetKernelArg(_kernel, 3, sizeof(cl_uint), &_d);
    size_t globalWorkSize[2] = { _m, _n };
    std::vector<cl_event> inputEvents = GetNodeInputEvents();
    cl_event ownEvent;
    OCLWrappers::CheckCLError(
        clEnqueueNDRangeKernel(_queue, _kernel, 2, nullptr, globalWorkSize, nullptr, inputEvents.size(), inputEvents.data(), &ownEvent)
    , "clEnqueueNDRangeKernel (for MatrixMultNodeGPUKernel)");
    SetEvent(ownEvent);
}
