#include "ConstMultNodeCPUKernel.hpp"

#include <assert.h>

ConstMultNodeCPUKernel::ConstMultNodeCPUKernel(const float* const memIn, float* const memOut, float factor, size_t size)
    : _memIn(memIn), _memOut(memOut), _factor(factor), _size(size)
{
    assert(_memIn != nullptr);
    assert(_memOut != nullptr);
}

ConstMultNodeCPUKernel::~ConstMultNodeCPUKernel() { }

void ConstMultNodeCPUKernel::Run()
{
    for (size_t i = 0; i < _size; ++i)
    {
        _memOut[i] = _factor * _memIn[i];
    }
}
