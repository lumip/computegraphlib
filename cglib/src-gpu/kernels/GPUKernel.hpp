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
    const size_t _numberCUs;
    const GPUKernel::ConstList _inputKernels;
    OCLWrappers::Event _event;
protected:
    const cl_kernel _kernel;
    const cl_command_queue _queue;
protected:
    std::vector<cl_event> GetNodeInputEvents() const;
public:
    GPUKernel(size_t numberOfCUs, cl_command_queue queue, cl_kernel kernel, const GPUKernel::ConstList& inputKernels);
    virtual ~GPUKernel();
    void SetEvent(cl_event event);
    virtual bool HasEvent() const;
    virtual cl_event GetEvent() const;
    virtual size_t GetPreferredWorkGroupMultiple() const;
    virtual size_t GetMaxWorkGroupSize() const;
    virtual std::pair<size_t, size_t> GetWorkSize(size_t workItemCount) const;
};

#endif
