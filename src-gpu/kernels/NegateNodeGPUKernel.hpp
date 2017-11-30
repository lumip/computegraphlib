#ifndef _NEGATE_NODE_GPU_KERNEL_HPP_
#define _NEGATE_NODE_GPU_KERNEL_HPP_

#include "Kernel.hpp"

class NegateNodeGPUKernel : public Kernel
{
private:
    const float* const _memIn;
    float* const _memOut;
    const size_t _size;
public:
    NegateNodeGPUKernel(const float* const memIn, float* const memOut, size_t size);
    virtual ~NegateNodeGPUKernel();
    void Run();
};

#endif
