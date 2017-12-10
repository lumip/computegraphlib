#include "ConstMultNodeGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"
#include "nodes/Node.hpp"
#include "../OCLWrappers.hpp"

const std::string ConstMultNodeGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float* memIn, __global float* memOut, float factor)
{
    uint id = get_global_id(0);
    memOut[id] = factor * memIn[id];
}
)==kernel==";

ConstMultNodeGPUKernel::ConstMultNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memIn, cl_mem memOut, float factor, size_t size)
    : GPUKernel(queue, compiler.CompileKernel(KernelSource), inputKernels), _memIn(memIn), _memOut(memOut), _factor(factor), _size(size)
{ }

ConstMultNodeGPUKernel::~ConstMultNodeGPUKernel() { }

void ConstMultNodeGPUKernel::Run()
{
    clSetKernelArg(_kernel, 0, sizeof(cl_mem), &_memIn);
    clSetKernelArg(_kernel, 1, sizeof(cl_mem), &_memOut);
    clSetKernelArg(_kernel, 2, sizeof(cl_float), &_factor);
    size_t globalWorkSize[1] = { _size };
    std::vector<cl_event> inputEvents = GetNodeInputEvents();
    cl_event ownEvent;
    OCLWrappers::CheckCLError(
        clEnqueueNDRangeKernel(_queue, _kernel, 1, nullptr, globalWorkSize, nullptr, inputEvents.size(), inputEvents.data(), &ownEvent)
    , "clEnqueueNDRangeKernel (for ConstMultNodeGPUKernel)");
    SetEvent(ownEvent);
}
