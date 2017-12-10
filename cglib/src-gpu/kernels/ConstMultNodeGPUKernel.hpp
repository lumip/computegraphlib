#ifndef _CONST_MULT_NODE_GPU_KERNEL_HPP_
#define _CONST_MULT_NODE_GPU_KERNEL_HPP_

#include <CL/cl.h>

#include "types.hpp"
#include "GPUKernel.hpp"

class OpenCLCompiler;

class ConstMultNodeGPUKernel : public GPUKernel
{
private:
    static const std::string KernelSource;
private:
    const cl_mem _memIn;
    const cl_mem _memOut;
    const float _factor;
    const size_t _size;
public:
    ConstMultNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memIn, cl_mem memOut, float factor, size_t size);
    virtual ~ConstMultNodeGPUKernel();
    void Run();
};

#endif
