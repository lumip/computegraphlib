#include "ExpFuncNodeGPUKernel.hpp"

#include <cmath>
#include <assert.h>

ExpFuncNodeGPUKernel::ExpFuncNodeGPUKernel(const float* const memIn, float* const memOut, size_t size)
    : _memIn(memIn), _memOut(memOut), _size(size)
{
    assert(_memIn != nullptr);
    assert(_memOut != nullptr);
}

ExpFuncNodeGPUKernel::~ExpFuncNodeGPUKernel() { }

void ExpFuncNodeGPUKernel::Run()
{
    for (size_t i = 0; i < _size; ++i)
    {
        _memOut[i] = std::exp(_memIn[i]);
    }
}
