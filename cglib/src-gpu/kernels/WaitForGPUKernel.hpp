#ifndef _WAIT_FOR_GPU_KERNEL_HPP_
#define _WAIT_FOR_GPU_KERNEL_HPP_

#include "GPUKernel.hpp"

class WaitForGPUKernel : public GPUKernel
{

public:
    WaitForGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, const GPUKernel::ConstList inputKernels);
    virtual ~WaitForGPUKernel();
    void Run();
};

#endif
