#ifndef _VECTOR_ADD_NODE_GPU_KERNEL_HPP_
#define _VECTOR_ADD_NODE_GPU_KERNEL_HPP_

#include <CL/cl.h>

#include "GPUKernel.hpp"

class OpenCLCompiler;

class VectorAddNodeGPUKernel : public GPUKernel
{
private:
    static const std::string KernelSource;
private:
    const cl_mem _memA; // invariant: _memA will always have output size (it is the unbroadcasted operand)
    const cl_mem _memB; // _memB may be a column or row vector that will be broadcasted over _memA
    const cl_mem _memRes;
    const MemoryDimensions _dimA;
    const MemoryDimensions _dimB;
public:
    VectorAddNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memA, cl_mem memB, cl_mem memRes, MemoryDimensions dimA, MemoryDimensions dimB);
    virtual ~VectorAddNodeGPUKernel();
    void Run();
};

#endif
