#ifndef _REDUCE_SUM_NODE_GPU_KERNEL_HPP_
#define _REDUCE_SUM_NODE_GPU_KERNEL_HPP_

#include <CL/cl.h>

#include "GPUKernel.hpp"

class OpenCLCompiler;

class ReduceSumNodeGPUKernel : public GPUKernel
{
private:
    static const std::string KernelSource;
private:
    const cl_mem _memIn;
    const cl_mem _memOut;
    const MemoryDimensions _dimIn;
    const size_t _axis;
public:
    ReduceSumNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memIn, cl_mem memOut, const MemoryDimensions dimIn, size_t axis);
    virtual ~ReduceSumNodeGPUKernel();
    void Run();
};

#endif
