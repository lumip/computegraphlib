#ifndef _REDUCE_MEAN_NODE_GPU_KERNEL_HPP_
#define _REDUCE_MEAN_NODE_GPU_KERNEL_HPP_

#include "Kernel.hpp"
#include "ReduceSumNodeGPUKernel.hpp"

class ReduceMeanNodeGPUKernel : public Kernel
{
private:
    float* const _memOut;
    ReduceSumNodeGPUKernel _sumKernel;
    const size_t _outSize;
    const float _reducedDimSize;
public:
    ReduceMeanNodeGPUKernel(const float* const memIn, float* const memOut, MemoryDimensions dimIn, size_t axis);
    virtual ~ReduceMeanNodeGPUKernel();
    void Run();
};

#endif
