#include "CopyDataGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"
#include "../OCLWrappers.hpp"

const std::string CopyDataGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float const * const memIn, __global float* const memOut, uint const offsetIn, uint const offsetOut, uint const strideIn, uint const strideOut, uint const maxId)
{
    uint i = get_global_id(0);
    if (i >= maxId) return;
    memOut[i * strideOut + offsetOut] = memIn[i * strideIn + offsetIn];
}
)==kernel==";

CopyDataGPUKernel::CopyDataGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memIn, cl_mem memOut, size_t count, size_t offsetIn, size_t offsetOut, size_t strideIn, size_t strideOut)
    : GPUKernel(queue, compiler.CompileKernel(KernelSource), inputKernels), _memIn(memIn), _memOut(memOut), _count(count), _offsetIn(offsetIn), _offsetOut(offsetOut), _strideIn(strideIn), _strideOut(strideOut)
{ }

CopyDataGPUKernel::~CopyDataGPUKernel() { }

void CopyDataGPUKernel::Run()
{
    clSetKernelArg(_kernel, 0, sizeof(cl_mem), &_memIn);
    clSetKernelArg(_kernel, 1, sizeof(cl_mem), &_memOut);
    clSetKernelArg(_kernel, 2, sizeof(cl_uint), &_offsetIn);
    clSetKernelArg(_kernel, 3, sizeof(cl_uint), &_offsetOut);
    clSetKernelArg(_kernel, 4, sizeof(cl_uint), &_strideIn);
    clSetKernelArg(_kernel, 5, sizeof(cl_uint), &_strideOut);
    clSetKernelArg(_kernel, 6, sizeof(cl_uint), &_count);
    std::pair<size_t, size_t> workSize = GetWorkSize(_count);
    std::vector<cl_event> inputEvents = GetNodeInputEvents();
    cl_event ownEvent;
    OCLWrappers::CheckCLError(
        clEnqueueNDRangeKernel(_queue, _kernel, 1, nullptr, &(workSize.first), &(workSize.second), inputEvents.size(), inputEvents.data(), &ownEvent)
    , "clEnqueueNDRangeKernel (for CopyDataGPUKernel)");
    SetEvent(ownEvent);
}
