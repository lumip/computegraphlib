#ifndef _CONST_MULT_NODE_GPU_KERNEL_HPP_
#define _CONST_MULT_NODE_GPU_KERNEL_HPP_

#include "Kernel.hpp"

class ConstMultNodeGPUKernel : public Kernel
{
private:
    const float* const _memIn;
    float* const _memOut;
    const float _factor;
    const size_t _size;
public:
    ConstMultNodeGPUKernel(const float* const memIn, float* const memOut, float factor, size_t size);
    virtual ~ConstMultNodeGPUKernel();
    void Run();
};

#endif
