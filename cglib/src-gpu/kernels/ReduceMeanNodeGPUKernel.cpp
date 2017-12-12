#include "ReduceMeanNodeGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"

ReduceMeanNodeGPUKernel::ReduceMeanNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memIn, cl_mem memOut, const MemoryDimensions dimIn, size_t axis)
    : GPUKernel(compiler.GetComputeUnitCount(), queue, 0, inputKernels)
    , _reducSumNode(compiler, queue, inputKernels, memIn, memOut, dimIn, axis)
    , _constDivNode(compiler, queue, inputKernels, memOut, memOut, 1.0f/static_cast<float>(dimIn.dims[axis]), dimIn.dims[1-axis])
{
    assert(axis == 0 || axis == 1);
}

ReduceMeanNodeGPUKernel::~ReduceMeanNodeGPUKernel() { }

void ReduceMeanNodeGPUKernel::Run()
{
    _reducSumNode.Run();
    _constDivNode.Run();
}

cl_event ReduceMeanNodeGPUKernel::GetEvent() const
{
    return _constDivNode.GetEvent();
}
