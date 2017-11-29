#ifndef _VECTOR_ADD_NODE_CPU_KERNEL_HPP_
#define _VECTOR_ADD_NODE_CPU_KERNEL_HPP_

#include "Kernel.hpp"

class VectorAddNodeCPUKernel : public Kernel
{
private:
    const float* const _memoryA;
    const float* const _memoryB;
    float* const _memoryResult;
    const size_t _size;
public:
    VectorAddNodeCPUKernel(const float* const memoryA, const float* const memoryB, float* const memoryResult, size_t size);
    virtual ~VectorAddNodeCPUKernel();
    void Run();
};

#endif
