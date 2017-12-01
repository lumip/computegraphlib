#include "ConstMultNodeGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"

const std::string ConstMultNodeGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float* memIn, __global float* memOut, float factor)
{
    uint id = get_global_id(0);
    memOut[id] = factor * memIn[id];
}
)==kernel==";

ConstMultNodeGPUKernel::ConstMultNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, cl_mem memIn, cl_mem memOut, float factor, size_t size)
    : _kernel(compiler.CompileKernel(KernelSource)), _queue(queue), _memIn(memIn), _memOut(memOut), _factor(factor), _size(size)
{ }

ConstMultNodeGPUKernel::~ConstMultNodeGPUKernel() { }

void ConstMultNodeGPUKernel::Run()
{
    clSetKernelArg(_kernel.get(), 0, sizeof(cl_mem), &_memIn);
    clSetKernelArg(_kernel.get(), 1, sizeof(cl_mem), &_memOut);
    clSetKernelArg(_kernel.get(), 2, sizeof(cl_float), &_factor);
    size_t globalWorkSize[1] = { _size };
    OCLWrappers::CheckCLError(
        clEnqueueNDRangeKernel(_queue, _kernel.get(), 1, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr)
    , "clEnqueueNDRangeKernel (for ConstMultNodeGPUKernel)");
}
