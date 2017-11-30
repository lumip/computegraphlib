#include "ReduceMeanNodeGPUKernel.hpp"

#include <assert.h>

ReduceMeanNodeGPUKernel::ReduceMeanNodeGPUKernel(const float* const memIn, float* const memOut, MemoryDimensions dimIn, size_t axis)
    : _memOut(memOut), _sumKernel(memIn, memOut, dimIn, axis), _outSize(dimIn.dims[1-axis]), _reducedDimSize(static_cast<float>(dimIn.dims[axis]))
{
    assert(_memOut != nullptr);
}

ReduceMeanNodeGPUKernel::~ReduceMeanNodeGPUKernel() { }

void ReduceMeanNodeGPUKernel::Run()
{
    _sumKernel.Run(); // sum up all entries along the given axis using the implementation in ReduceSumNodeCPUKernel

    // divide all sums to get the mean
    for (size_t i = 0; i < _outSize; ++i)
    {
        _memOut[i] /= _reducedDimSize;
    }
}
