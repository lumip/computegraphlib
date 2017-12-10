#include "GPUKernel.hpp"

#include "nodes/Node.hpp"
#include "OpenCLCompiler.hpp"

GPUKernel::GPUKernel(cl_command_queue queue, cl_kernel kernel, const GPUKernel::ConstList& inputKernels)
    : _inputKernels(inputKernels)
    , _event()
    , _kernel(kernel)
    , _queue(queue)
{ }

GPUKernel::~GPUKernel() { }

std::vector<cl_event> GPUKernel::GetNodeInputEvents() const
{
    std::vector<cl_event> events;
    events.reserve(_inputKernels.size());
    for (GPUKernel const * inputKernel : _inputKernels)
    {
        if (inputKernel != nullptr)
        {
            events.push_back(inputKernel->GetEvent());
        }
    }
    return events;
}

void GPUKernel::SetEvent(cl_event event)
{
    _event = OCLWrappers::Event(event);
}

cl_event GPUKernel::GetEvent() const
{
    return _event.get();
}
