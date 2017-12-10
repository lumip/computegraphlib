#ifndef _EXP_FUNC_NODE_GPU_KERNEL_HPP_
#define _EXP_FUNC_NODE_GPU_KERNEL_HPP_

#include <CL/cl.h>

#include "Kernel.hpp"

class OpenCLCompiler;

class ExpFuncNodeGPUKernel : public Kernel
{
private:
    static const std::string KernelSource;
private:
    const cl_kernel _kernel;
    const cl_command_queue _queue;
    const cl_mem _memIn;
    const cl_mem _memOut;
    const size_t _size;
public:
    ExpFuncNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, cl_mem memIn, cl_mem memOut,  size_t size);
    ~ExpFuncNodeGPUKernel();
    void Run();
};

#endif
