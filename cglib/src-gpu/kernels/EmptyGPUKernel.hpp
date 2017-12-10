#ifndef _EMPTY_GPU_KERNEL_HPP_
#define _EMPTY_GPU_KERNEL_HPP_

#include "GPUKernel.hpp"

class EmptyGPUKernel : public GPUKernel
{
public:
    EmptyGPUKernel(cl_command_queue queue, const GPUKernel::ConstList& inputKernels);
    virtual ~EmptyGPUKernel();
    void Run();
};

#endif
