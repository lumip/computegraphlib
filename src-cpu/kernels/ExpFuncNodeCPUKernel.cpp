#include "ExpFuncNodeCPUKernel.hpp"

#include <cmath>
#include <assert.h>

ExpFuncNodeCPUKernel::ExpFuncNodeCPUKernel(const float* const memIn, float* const memOut, size_t size)
    : _memIn(memIn), _memOut(memOut), _size(size)
{
    assert(_memIn != nullptr);
    assert(_memOut != nullptr);
}

ExpFuncNodeCPUKernel::~ExpFuncNodeCPUKernel() { }

void ExpFuncNodeCPUKernel::Run()
{
    for (size_t i = 0; i < _size; ++i)
    {
        _memOut[i] = std::exp(_memIn[i]);
    }
}
