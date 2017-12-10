#include "NegateNodeGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"
#include "../OCLWrappers.hpp"

const std::string NegateNodeGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float* memIn, __global float* memOut)
{
    uint i = get_global_id(0);
    memOut[i] = -memIn[i];
}
)==kernel==";

NegateNodeGPUKernel::NegateNodeGPUKernel(OpenCLCompiler& compiler, const cl_command_queue queue, cl_mem memIn, cl_mem memOut,  size_t size)
    : _kernel(compiler.CompileKernel(KernelSource)), _queue(queue), _memIn(memIn), _memOut(memOut), _size(size)
{ }

NegateNodeGPUKernel::~NegateNodeGPUKernel() { }

void NegateNodeGPUKernel::Run()
{
    clSetKernelArg(_kernel, 0, sizeof(cl_mem), &_memIn);
    clSetKernelArg(_kernel, 1, sizeof(cl_mem), &_memOut);
    size_t globalWorkSize[1] = { _size };
    OCLWrappers::CheckCLError(
        clEnqueueNDRangeKernel(_queue, _kernel, 1, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr)
    , "clEnqueueNDRangeKernel (for NegateNodeGPUKernel)");
}
