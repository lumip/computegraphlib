#include "ExpFuncNodeCPUKernel.hpp"

#include <cmath>

ExpFuncNodeCPUKernel::ExpFuncNodeCPUKernel(const float* const memIn, float* const memOut, size_t size)
    : _memIn(memIn), _memOut(memOut), _size(size)
{ }

ExpFuncNodeCPUKernel::~ExpFuncNodeCPUKernel() { }

void ExpFuncNodeCPUKernel::Run()
{
    for (size_t i = 0; i < _size; ++i)
    {
        _memOut[i] = std::exp(_memIn[i]);
    }
}
