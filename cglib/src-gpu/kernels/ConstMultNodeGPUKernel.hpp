#ifndef _CONST_MULT_NODE_GPU_KERNEL_HPP_
#define _CONST_MULT_NODE_GPU_KERNEL_HPP_

#include "Kernel.hpp"

#include "../OCLWrappers.hpp"

class OpenCLCompiler;

class ConstMultNodeGPUKernel : public Kernel
{
private:
    static const std::string KernelSource;
private:
    OCLWrappers::Kernel _kernel;
    const cl_command_queue _queue;
    const cl_mem _memIn;
    const cl_mem _memOut;
    const float _factor;
    const size_t _size;
public:
    ConstMultNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, cl_mem memIn, cl_mem memOut, float factor, size_t size);
    virtual ~ConstMultNodeGPUKernel();
    void Run();
};

#endif
