#ifndef _NEGATE_NODE_GPU_KERNEL_HPP_
#define _NEGATE_NODE_GPU_KERNEL_HPP_

#include <CL/cl.h>

#include "GPUKernel.hpp"

class OpenCLCompiler;

class NegateNodeGPUKernel : public GPUKernel
{
private:
    static const std::string KernelSource;
private:
    const cl_mem _memIn;
    const cl_mem _memOut;
    const size_t _size;
public:
    NegateNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memIn, cl_mem memOut,  size_t size);
    virtual ~NegateNodeGPUKernel();
    void Run();
};

#endif
