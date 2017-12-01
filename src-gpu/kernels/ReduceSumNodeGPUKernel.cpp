#include "ReduceSumNodeGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"

const std::string ReduceSumNodeGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float* memIn, __global float* memOut, uint offsetStride, uint sumStride, uint count)
{
    size_t id = get_global_id(0);
    float sum = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        sum += memIn[id * offsetStride + i * sumStride];
    }
    memOut[id] = sum;
}
)==kernel==";

ReduceSumNodeGPUKernel::ReduceSumNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, cl_mem memIn, cl_mem memOut, const MemoryDimensions dimIn, size_t axis)
    : _kernel(compiler.CompileKernel(KernelSource)), _queue(queue), _memIn(memIn), _memOut(memOut), _dimIn(dimIn), _axis(axis)
{
    assert(_axis == 0 || _axis == 1);
}

ReduceSumNodeGPUKernel::~ReduceSumNodeGPUKernel() { }

void ReduceSumNodeGPUKernel::Run()
{
    // we will have one work item for each result element computing one sum along the given axis
    // default case, axis == 0
    cl_uint offsetStride = 1;   // the stride distance between the first cells for each work item
    cl_uint sumStride = static_cast<cl_uint>(_dimIn.xDim); // the stride distance with which each work item progresses through the input
    cl_uint workersCount = static_cast<cl_uint>(_dimIn.xDim); // the number of work items (i.e. the size of the output)
    cl_uint workSize = static_cast<cl_uint>(_dimIn.yDim); // the number of cells each work item will process
    if (_axis == 1)
    {
        std::swap(offsetStride, sumStride);
        std::swap(workersCount, workSize);
    }

    clSetKernelArg(_kernel.get(), 0, sizeof(cl_mem), &_memIn);
    clSetKernelArg(_kernel.get(), 1, sizeof(cl_mem), &_memOut);
    clSetKernelArg(_kernel.get(), 2, sizeof(cl_uint), &offsetStride);
    clSetKernelArg(_kernel.get(), 3, sizeof(cl_uint), &sumStride);
    clSetKernelArg(_kernel.get(), 4, sizeof(cl_uint), &workSize);
    size_t globalWorkSize[2] = { workersCount };
    OCLWrappers::CheckCLError(
        clEnqueueNDRangeKernel(_queue, _kernel.get(), 1, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr)
    , "clEnqueueNDRangeKernel (for ReduceSumNodeGPUKernel)");
}
