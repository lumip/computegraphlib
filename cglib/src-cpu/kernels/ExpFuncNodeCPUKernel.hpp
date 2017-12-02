#ifndef _EXP_FUNC_NODE_CPU_KERNEL_HPP_
#define _EXP_FUNC_NODE_CPU_KERNEL_HPP_

#include "Kernel.hpp"

class ExpFuncNodeCPUKernel : public Kernel
{
private:
    const float* const _memIn;
    float* const _memOut;
    const size_t _size;
public:
    ExpFuncNodeCPUKernel(const float* const memIn, float* const memOut, size_t size);
    ~ExpFuncNodeCPUKernel();
    void Run();
};

#endif
