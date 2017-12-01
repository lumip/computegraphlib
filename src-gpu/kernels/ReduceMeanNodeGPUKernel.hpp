#ifndef _REDUCE_MEAN_NODE_GPU_KERNEL_HPP_
#define _REDUCE_MEAN_NODE_GPU_KERNEL_HPP_

#include <memory>

#include "Kernel.hpp"

#include "../OCLWrappers.hpp"
#include "ConstMultNodeGPUKernel.hpp"
#include "ReduceSumNodeGPUKernel.hpp"

class OpenCLCompiler;

class ReduceMeanNodeGPUKernel : public Kernel
{
private:
    ReduceSumNodeGPUKernel _reducSumNode;
    ConstMultNodeGPUKernel _constDivNode;
public:
    ReduceMeanNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, cl_mem memIn, cl_mem memOut, const MemoryDimensions dimIn, size_t axis);
    virtual ~ReduceMeanNodeGPUKernel();
    void Run();
};

#endif
