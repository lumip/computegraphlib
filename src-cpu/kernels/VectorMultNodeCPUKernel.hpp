#ifndef _VECTOR_MULT_NODE_CPU_KERNEL_HPP_
#define _VECTOR_MULT_NODE_CPU_KERNEL_HPP_

#include "Kernel.hpp"

class VectorMultNodeCPUKernel : public Kernel
{
private:
    const float* const _memA;
    const float* const _memB;
    float* const _memRes;
    const size_t _size;
public:
    VectorMultNodeCPUKernel(const float* const memA, const float* const memB, float* const memRes, size_t size);
    virtual ~VectorMultNodeCPUKernel();
    void Run();
};

#endif
