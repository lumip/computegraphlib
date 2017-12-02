#ifndef _MATRIX_MULT_NODE_CPU_KERNEL_HPP_
#define _MATRIX_MULT_NODE_CPU_KERNEL_HPP_

#include "Kernel.hpp"

class MatrixMultNodeCPUKernel : public Kernel
{
private:
    const float* const _memA;
    const float* const _memB;
    float* const _memRes;
    const size_t _m;
    const size_t _n;
    const size_t _d;
private:
    inline size_t GetIndex(size_t i, size_t j, size_t stride) const
    {
        return i * stride + j;
    }
public:
    MatrixMultNodeCPUKernel(const float* const memA, const float* const memB, float* const memRes, const size_t m, const size_t n, const size_t d);
    virtual ~MatrixMultNodeCPUKernel();
    void Run();
};

#endif
