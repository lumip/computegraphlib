#ifndef _GPU_KERNEL_HPP_
#define _GPU_KERNEL_HPP_

#include "Kernel.hpp"
#include "../OCLWrappers.hpp"

class OpenCLCompiler;

class GPUKernel : public Kernel
{
public:
    typedef std::vector<GPUKernel const *> ConstList;
private:
    const GPUKernel::ConstList _inputKernels;
    OCLWrappers::Event _event;
protected:
    const cl_kernel _kernel;
    const cl_command_queue _queue;
protected:
    std::vector<cl_event> GetNodeInputEvents() const;
    void SetEvent(cl_event event);
public:
    GPUKernel(cl_command_queue queue, cl_kernel kernel, const GPUKernel::ConstList& inputKernels);
    virtual ~GPUKernel();
    virtual cl_event GetEvent() const;
};

#endif
