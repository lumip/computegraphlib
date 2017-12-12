#include "GPUKernel.hpp"

#include <cmath>

#include "nodes/Node.hpp"
#include "OpenCLCompiler.hpp"

GPUKernel::GPUKernel(size_t numberOfCUs, cl_command_queue queue, cl_kernel kernel, const GPUKernel::ConstList& inputKernels)
    : _numberCUs(numberOfCUs)
    , _inputKernels(inputKernels)
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

size_t GPUKernel::GetPreferredWorkGroupMultiple() const
{
    size_t workGroupSize = 0;
    OCLWrappers::CheckCLError(
        clGetKernelWorkGroupInfo(_kernel, nullptr, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &workGroupSize, nullptr)
    , "clGetKernelWorkGroupInfo(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE) (for GPUKernel::GetPreferredWorkGroupSize())");
    return workGroupSize;
}

size_t GPUKernel::GetMaxWorkGroupSize() const
{
    size_t workGroupSize = 0;
    OCLWrappers::CheckCLError(
        clGetKernelWorkGroupInfo(_kernel, nullptr, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workGroupSize, nullptr)
    , "clGetKernelWorkGroupInfo(CL_KERNEL_WORK_GROUP_SIZE) (for GPUKernel::GetPreferredWorkGroupSize())");
    return workGroupSize;
}

std::pair<size_t, size_t> GPUKernel::GetWorkSize(size_t workItemCount) const
{
    size_t preferredMultiple = GetPreferredWorkGroupMultiple();

    size_t usedCus = std::min((workItemCount / (GetMaxWorkGroupSize() / 2)) + 1, _numberCUs);
    size_t workGroupsPerCU = ((workItemCount/(usedCus * GetMaxWorkGroupSize())) + 1);
    size_t workGroupCount = (workGroupsPerCU * usedCus);
    size_t workGroupSize = workItemCount / workGroupCount;
    if (workGroupSize % preferredMultiple != 0)
    {
        workGroupSize = ((workGroupSize / preferredMultiple) + 1) * preferredMultiple; // adjust total work size to be the next largest preferred multiple
    }
    workItemCount = std::max(static_cast<size_t>(1), workGroupCount * workGroupSize); // ensure that the size is at least 1

    return std::pair<size_t, size_t>(workItemCount, workGroupSize);
}
