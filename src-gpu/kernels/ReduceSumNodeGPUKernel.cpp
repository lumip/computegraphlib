#include "ReduceSumNodeGPUKernel.hpp"

#include <assert.h>

ReduceSumNodeGPUKernel::ReduceSumNodeGPUKernel(const float* const memIn, float* const memOut, MemoryDimensions dimIn, size_t axis)
    : _memIn(memIn), _memOut(memOut), _dimIn(dimIn), _axis(axis)
{
    assert(_memIn != nullptr);
    assert(_memOut != nullptr);
    assert(_axis == 0 || _axis == 1);
}

ReduceSumNodeGPUKernel::~ReduceSumNodeGPUKernel() { }

void ReduceSumNodeGPUKernel::Run()
{
    const size_t outSize = _dimIn.dims[1-_axis];
    std::fill_n(_memOut, outSize, 0.0f);
    const size_t rowIdMask = 0 - _axis; // mask of all 0 iff _axis==0, all 1 iff _axis==1 (all bits 1 achieved by setting value to -1)
    const size_t colIdMask = _axis - 1; // mask of all 1 iff _axis==0, all 1 iff _axis==1
    assert(rowIdMask != colIdMask);
    for (size_t i = 0; i < _dimIn.yDim; ++i)
    {
        for (size_t j = 0; j < _dimIn.xDim; ++j) // always let the inner loop iterate over columns first for cache friendlieness
        {
            size_t outId = (i & rowIdMask) | (j & colIdMask); // select whether the current value goes into row or column bin
            _memOut[outId] += _memIn[GetIndex(i, j, _dimIn.xDim)];
        }
    }
}
