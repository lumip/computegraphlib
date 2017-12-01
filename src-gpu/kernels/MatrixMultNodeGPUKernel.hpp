#ifndef _MATRIX_MULT_NODE_GPU_KERNEL_HPP_
#define _MATRIX_MULT_NODE_GPU_KERNEL_HPP_

#include "Kernel.hpp"

#include "../OCLWrappers.hpp"

class OpenCLCompiler;

class MatrixMultNodeGPUKernel : public Kernel
{
private:
    static const std::string KernelSource;
private:
    const OCLWrappers::Kernel _kernel;
    const cl_command_queue _queue;
    const cl_mem _memA;
    const cl_mem _memB;
    const cl_mem _memRes;
    const size_t _m;
    const size_t _n;
    const size_t _d;
public:
    MatrixMultNodeGPUKernel(OpenCLCompiler& compiler, const cl_command_queue queue, cl_mem memA, cl_mem memB, cl_mem memRes, size_t m, size_t n, size_t d);
    virtual ~MatrixMultNodeGPUKernel();
    void Run();
};

#endif
