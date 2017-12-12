#ifndef _REDUCE_MEAN_NODE_GPU_KERNEL_HPP_
#define _REDUCE_MEAN_NODE_GPU_KERNEL_HPP_

#include <memory>
#include <CL/cl.h>

#include "GPUKernel.hpp"
#include "types.hpp"

#include "ConstMultNodeGPUKernel.hpp"
#include "ReduceSumNodeGPUKernel.hpp"

class OpenCLCompiler;

class ReduceMeanNodeGPUKernel : public GPUKernel
{
private:
    ReduceSumNodeGPUKernel _reducSumNode;
    ConstMultNodeGPUKernel _constDivNode;
public:
    ReduceMeanNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memIn, cl_mem memOut, const MemoryDimensions dimIn, size_t axis);
    virtual ~ReduceMeanNodeGPUKernel();
    void Run();
    cl_event GetEvent() const;
};

#endif
