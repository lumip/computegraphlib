#include "EmptyGPUKernel.hpp"

#include "OpenCLCompiler.hpp"

EmptyGPUKernel::EmptyGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, const GPUKernel::ConstList& inputKernels)
    : GPUKernel(compiler.GetComputeUnitCount(), queue, 0, inputKernels)
{ }

EmptyGPUKernel::~EmptyGPUKernel() { }

void EmptyGPUKernel::Run()
{
    cl_event ownEvent;
    std::vector<cl_event> inputEvents = GetNodeInputEvents();
    cl_event* rawEventList = nullptr;
    if (inputEvents.size() > 0)
    {
        rawEventList = inputEvents.data();
    }
    OCLWrappers::CheckCLError(
        clEnqueueMarkerWithWaitList(_queue, inputEvents.size(), rawEventList, &ownEvent)
    , "clEnqueueMarkerWithWaitList (for EmptyGPUKernel)");
    SetEvent(ownEvent);
}
