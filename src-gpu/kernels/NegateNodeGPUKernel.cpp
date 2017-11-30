#include "NegateNodeGPUKernel.hpp"

#include <assert.h>

NegateNodeGPUKernel::NegateNodeGPUKernel(const float* const memIn, float* const memOut, size_t size)
    : _memIn(memIn), _memOut(memOut), _size(size)
{
    assert(_memIn != nullptr);
    assert(_memOut != nullptr);
}

NegateNodeGPUKernel::~NegateNodeGPUKernel() { }

void NegateNodeGPUKernel::Run()
{
    for (size_t i = 0; i < _size; ++i)
    {
        _memOut[i] = -_memIn[i];
    }
}
