#ifndef _EXP_FUNC_NODE_GPU_KERNEL_HPP_
#define _EXP_FUNC_NODE_GPU_KERNEL_HPP_

#include "Kernel.hpp"

class ExpFuncNodeGPUKernel : public Kernel
{
private:
    const float* const _memIn;
    float* const _memOut;
    const size_t _size;
public:
    ExpFuncNodeGPUKernel(const float* const memIn, float* const memOut, size_t size);
    ~ExpFuncNodeGPUKernel();
    void Run();
};

#endif
