#include "NegateNodeCPUKernel.hpp"

#include <assert.h>

NegateNodeCPUKernel::NegateNodeCPUKernel(const float* const memIn, float* const memOut, size_t size)
    : _memIn(memIn), _memOut(memOut), _size(size)
{
    assert(_memIn != nullptr);
    assert(_memOut != nullptr);
}

NegateNodeCPUKernel::~NegateNodeCPUKernel() { }

void NegateNodeCPUKernel::Run()
{
    for (size_t i = 0; i < _size; ++i)
    {
        _memOut[i] = -_memIn[i];
    }
}
