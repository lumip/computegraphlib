#ifndef _VECTOR_DIV_NODE_GPU_KERNEL_HPP_
#define _VECTOR_DIV_NODE_GPU_KERNEL_HPP_

#include "Kernel.hpp"

class VectorDivNodeGPUKernel : public Kernel
{
private:
    const float* const _memA; // invariant: _memA will always have output size (it is the unbroadcasted operand)
    const float* const _memB; // _memB may be a column or row vector that will be broadcasted over _memA
    float* const _memRes;
    const MemoryDimensions _dimA;
    const MemoryDimensions _dimB;
private:
    inline size_t GetIndex(size_t i, size_t j, size_t stride) const
    {
        return i * stride + j;
    }
public:
    VectorDivNodeGPUKernel(const float* const memA, const float* const memB, float* const memRes, MemoryDimensions dimA, MemoryDimensions dimB);
    virtual ~VectorDivNodeGPUKernel();
    void Run();
};

#endif
