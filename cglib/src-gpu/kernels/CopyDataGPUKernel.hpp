#ifndef _COPY_DATA_GPU_KERNEL_HPP_
#define _COPY_DATA_GPU_KERNEL_HPP_

#include <CL/cl.h>

#include "GPUKernel.hpp"

class OpenCLCompiler;

class CopyDataGPUKernel : public GPUKernel
{
private:
    static const std::string KernelSource;
private:
    const cl_mem _memIn;
    const cl_mem _memOut;
    const size_t _count;
    const size_t _offsetIn;
    const size_t _offsetOut;
    const size_t _strideIn;
    const size_t _strideOut;
public:
    CopyDataGPUKernel(OpenCLCompiler& compiler,
                      cl_command_queue queue,
                      const GPUKernel::ConstList& inputKernels,
                      cl_mem memIn,
                      cl_mem memOut,
                      size_t count,
                      size_t offsetIn,
                      size_t offsetOut,
                      size_t strideIn,
                      size_t strideOut);
    virtual ~CopyDataGPUKernel();
    void Run();
};

#endif
