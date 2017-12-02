#ifndef _LOG_FUNC_NODE_CPU_KERNEL_HPP_
#define _LOG_FUNC_NODE_CPU_KERNEL_HPP_

#include "Kernel.hpp"

class LogFuncNodeCPUKernel : public Kernel
{
private:
    const float* const _memIn;
    float* const _memOut;
    const size_t _size;
public:
    LogFuncNodeCPUKernel(const float* const memIn, float* const memOut, size_t size);
    virtual ~LogFuncNodeCPUKernel();
    void Run();
};

#endif
