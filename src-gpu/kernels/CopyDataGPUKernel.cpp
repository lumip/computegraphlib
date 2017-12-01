#include "CopyDataGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"

const std::string CopyDataGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float* memIn, __global float* memOut, uint offsetIn, uint offsetOut, uint strideIn, uint strideOut)
{
    uint i = get_global_id(0);
    memOut[i * strideOut + offsetOut] = memIn[i * strideIn + offsetIn];
}
)==kernel==";

CopyDataGPUKernel::CopyDataGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, cl_mem memIn, cl_mem memOut, size_t count, size_t offsetIn, size_t offsetOut, size_t strideIn, size_t strideOut)
    : _kernel(compiler.CompileKernel(KernelSource)), _queue(queue), _memIn(memIn), _memOut(memOut), _count(count), _offsetIn(offsetIn), _offsetOut(offsetOut), _strideIn(strideIn), _strideOut(strideOut)
{ }

CopyDataGPUKernel::~CopyDataGPUKernel() { }

void CopyDataGPUKernel::Run()
{
    clSetKernelArg(_kernel.get(), 0, sizeof(cl_mem), &_memIn);
    clSetKernelArg(_kernel.get(), 1, sizeof(cl_mem), &_memOut);
    clSetKernelArg(_kernel.get(), 2, sizeof(cl_uint), &_offsetIn);
    clSetKernelArg(_kernel.get(), 3, sizeof(cl_uint), &_offsetOut);
    clSetKernelArg(_kernel.get(), 4, sizeof(cl_uint), &_strideIn);
    clSetKernelArg(_kernel.get(), 5, sizeof(cl_uint), &_strideOut);
    size_t globalWorkSize[1] = { _count };
    OCLWrappers::CheckCLError(
        clEnqueueNDRangeKernel(_queue, _kernel.get(), 1, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr)
    , "clEnqueueNDRangeKernel (for CopyDataGPUKernel)");
}
