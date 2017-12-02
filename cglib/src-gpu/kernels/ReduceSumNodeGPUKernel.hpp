#ifndef _REDUCE_SUM_NODE_GPU_KERNEL_HPP_
#define _REDUCE_SUM_NODE_GPU_KERNEL_HPP_

#include "Kernel.hpp"

#include "../OCLWrappers.hpp"

class OpenCLCompiler;

class ReduceSumNodeGPUKernel : public Kernel
{
private:
    static const std::string KernelSource;
private:
    const OCLWrappers::Kernel _kernel;
    const cl_command_queue _queue;
    const cl_mem _memIn;
    const cl_mem _memOut;
    const MemoryDimensions _dimIn;
    const size_t _axis;
public:
    ReduceSumNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, cl_mem memIn, cl_mem memOut, const MemoryDimensions dimIn, size_t axis);
    virtual ~ReduceSumNodeGPUKernel();
    void Run();
};

#endif
