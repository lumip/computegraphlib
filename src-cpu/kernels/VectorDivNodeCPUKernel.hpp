#ifndef _VECTOR_DIV_NODE_CPU_KERNEL_HPP_
#define _VECTOR_DIV_NODE_CPU_KERNEL_HPP_

#include "Kernel.hpp"

class VectorDivNodeCPUKernel : public Kernel
{
private:
    const float* const _memA;
    const float* const _memB;
    float* const _memRes;
    const size_t _size;
public:
    VectorDivNodeCPUKernel(const float* const memA, const float* const memB, float* const memRes, size_t size);
    virtual ~VectorDivNodeCPUKernel();
    void Run();
};

#endif
