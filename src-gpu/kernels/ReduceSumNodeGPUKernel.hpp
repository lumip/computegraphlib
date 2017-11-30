#ifndef _REDUCE_SUM_NODE_GPU_KERNEL_HPP_
#define _REDUCE_SUM_NODE_GPU_KERNEL_HPP_

#include "Kernel.hpp"

class ReduceSumNodeGPUKernel : public Kernel
{
private:
    const float* const _memIn;
    float* const _memOut;
    const MemoryDimensions _dimIn;
    const size_t _axis;
private:
    inline size_t GetIndex(size_t i, size_t j, size_t stride) const
    {
        return i * stride + j;
    }
public:
    ReduceSumNodeGPUKernel(const float* const memIn, float* const memOut, MemoryDimensions dimIn, size_t axis);
    virtual ~ReduceSumNodeGPUKernel();
    void Run();
};

#endif
