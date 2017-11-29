#ifndef _NEGATE_NODE_CPU_KERNEL_HPP_
#define _NEGATE_NODE_CPU_KERNEL_HPP_

#include "Kernel.hpp"

class NegateNodeCPUKernel : public Kernel
{
private:
    const float* const _memIn;
    float* const _memOut;
    const size_t _size;
public:
    NegateNodeCPUKernel(const float* const memIn, float* const memOut, size_t size);
    virtual ~NegateNodeCPUKernel();
    void Run();
};

#endif
