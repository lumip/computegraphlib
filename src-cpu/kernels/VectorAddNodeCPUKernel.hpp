#ifndef _VECTOR_ADD_NODE_CPU_KERNEL_HPP_
#define _VECTOR_ADD_NODE_CPU_KERNEL_HPP_

#include "Kernel.hpp"

class VectorAddNodeCPUKernel : public Kernel
{
private:
    const float* const _memoryA; // invariant: _memoryA will always have output size (it is the unbroadcasted operand)
    const float* const _memoryB;
    float* const _memoryResult;
    const MemoryDimensions _dimA;
    const MemoryDimensions _dimB;
private:
    inline size_t GetIndex(size_t i, size_t j, size_t stride) const
    {
        return i * stride + j;
    }
public:
    VectorAddNodeCPUKernel(const float* const memoryA, const float* const memoryB, float* const memoryResult, MemoryDimensions dimA, MemoryDimensions dimB);
    virtual ~VectorAddNodeCPUKernel();
    void Run();
};

#endif
