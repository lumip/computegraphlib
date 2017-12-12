#include "ConstMultNodeGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"
#include "nodes/Node.hpp"
#include "../OCLWrappers.hpp"

const std::string ConstMultNodeGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float const * const memIn, __global float* const memOut, float const factor, uint const maxId)
{
    uint id = get_global_id(0);
    if (id >= maxId) return;
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
    clSetKernelArg(_kernel, 3, sizeof(cl_uint), &_size);
    std::pair<size_t, size_t> workSize = GetWorkSize(_size);
    std::vector<cl_event> inputEvents = GetNodeInputEvents();
    cl_event ownEvent;
    OCLWrappers::CheckCLError(
        clEnqueueNDRangeKernel(_queue, _kernel, 1, nullptr, &(workSize.first), &(workSize.second), inputEvents.size(), inputEvents.data(), &ownEvent)
    , "clEnqueueNDRangeKernel (for ConstMultNodeGPUKernel)");
    SetEvent(ownEvent);
}
