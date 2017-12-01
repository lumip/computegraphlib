#include "ReduceMeanNodeGPUKernel.hpp"

#include <assert.h>

ReduceMeanNodeGPUKernel::ReduceMeanNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, cl_mem memIn, cl_mem memOut, const MemoryDimensions dimIn, size_t axis)
    : _reducSumNode(compiler, queue, memIn, memOut, dimIn, axis)
    , _constDivNode(compiler, queue, memOut, memOut, 1.0f/static_cast<float>(dimIn.dims[axis]), dimIn.dims[1-axis])
{
    assert(axis == 0 || axis == 1);
}

ReduceMeanNodeGPUKernel::~ReduceMeanNodeGPUKernel() { }

void ReduceMeanNodeGPUKernel::Run()
{
    _reducSumNode.Run();
    _constDivNode.Run();
}
