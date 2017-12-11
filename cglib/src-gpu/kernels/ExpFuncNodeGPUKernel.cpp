#include "ExpFuncNodeGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"
#include "../OCLWrappers.hpp"

const std::string ExpFuncNodeGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float* memIn, __global float* memOut, uint maxId)
{
    uint id = get_global_id(0);
    if (id >= maxId) return;
    memOut[id] = exp(memIn[id]);
}
)==kernel==";

ExpFuncNodeGPUKernel::ExpFuncNodeGPUKernel(OpenCLCompiler& compiler, const cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memIn, cl_mem memOut,  size_t size)
    : GPUKernel(queue, compiler.CompileKernel(KernelSource), inputKernels), _memIn(memIn), _memOut(memOut), _size(size)
{ }

ExpFuncNodeGPUKernel::~ExpFuncNodeGPUKernel() { }

void ExpFuncNodeGPUKernel::Run()
{
    clSetKernelArg(_kernel, 0, sizeof(cl_mem), &_memIn);
    clSetKernelArg(_kernel, 1, sizeof(cl_mem), &_memOut);
    clSetKernelArg(_kernel, 2, sizeof(cl_uint), &_size);
    std::pair<size_t, size_t> workSize = GetWorkSize(_size);
    std::vector<cl_event> inputEvents = GetNodeInputEvents();
    cl_event ownEvent;
    OCLWrappers::CheckCLError(
        clEnqueueNDRangeKernel(_queue, _kernel, 1, nullptr, &(workSize.first), &(workSize.second), inputEvents.size(), inputEvents.data(), &ownEvent)
    , "clEnqueueNDRangeKernel (for ExpFuncNodeGPUKernel)");
    SetEvent(ownEvent);
}
