#ifndef _LOG_FUNC_NODE_GPU_KERNEL_HPP_
#define _LOG_FUNC_NODE_GPU_KERNEL_HPP_

#include "Kernel.hpp"

class LogFuncNodeGPUKernel : public Kernel
{
private:
    const float* const _memIn;
    float* const _memOut;
    const size_t _size;
public:
    LogFuncNodeGPUKernel(const float* const memIn, float* const memOut, size_t size);
    virtual ~LogFuncNodeGPUKernel();
    void Run();
};

#endif
