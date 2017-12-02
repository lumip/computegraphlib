#ifndef _CONST_MULT_NODE_CPU_KERNEL_HPP_
#define _CONST_MULT_NODE_CPU_KERNEL_HPP_

#include "Kernel.hpp"

class ConstMultNodeCPUKernel : public Kernel
{
private:
    const float* const _memIn;
    float* const _memOut;
    const float _factor;
    const size_t _size;
public:
    ConstMultNodeCPUKernel(const float* const memIn, float* const memOut, float factor, size_t size);
    virtual ~ConstMultNodeCPUKernel();
    void Run();
};

#endif
