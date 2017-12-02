#ifndef _COPY_DATA_CPU_KERNEL_HPP_
#define _COPY_DATA_CPU_KERNEL_HPP_

#include "Kernel.hpp"

class CopyDataCPUKernel : public Kernel
{
private:
    const float* const _memIn;
    float* const _memOut;
    const size_t _count;
    const size_t _strideIn;
    const size_t _strideOut;
public:
    CopyDataCPUKernel(const float* const memIn, float* const memOut, size_t count, size_t strideIn, size_t strideOut);
    virtual ~CopyDataCPUKernel();
    void Run();
};

#endif
