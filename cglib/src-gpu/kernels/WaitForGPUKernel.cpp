#include "WaitForGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"

WaitForGPUKernel::WaitForGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, const GPUKernel::ConstList inputKernels)
    : GPUKernel(compiler.GetComputeUnitCount(), queue, 0, inputKernels)
{ }

WaitForGPUKernel::~WaitForGPUKernel() { }

void WaitForGPUKernel::Run()
{
    std::vector<cl_event> inputEvents = GetNodeInputEvents();
    assert(inputEvents.size() > 0);
    cl_event ownEvent;
    OCLWrappers::CheckCLError(
        clEnqueueBarrierWithWaitList(_queue, inputEvents.size(), inputEvents.data(), &ownEvent)
    , "clEnqueueBarrierWithWaitList (for WaitForGPUKernel)");
    SetEvent(ownEvent);
}
