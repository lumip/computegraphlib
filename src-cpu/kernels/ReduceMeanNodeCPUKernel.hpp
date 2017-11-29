#ifndef _REDUCE_MEAN_NODE_CPU_KERNEL_HPP_
#define _REDUCE_MEAN_NODE_CPU_KERNEL_HPP_

#include "Kernel.hpp"
#include "ReduceSumNodeCPUKernel.hpp"

class ReduceMeanNodeCPUKernel : public Kernel
{
private:
    float* const _memOut;
    ReduceSumNodeCPUKernel _sumKernel;
    const size_t _outSize;
    const float _reducedDimSize;
public:
    ReduceMeanNodeCPUKernel(const float* const memIn, float* const memOut, MemoryDimensions dimIn, size_t axis);
    virtual ~ReduceMeanNodeCPUKernel();
    void Run();
};

#endif
