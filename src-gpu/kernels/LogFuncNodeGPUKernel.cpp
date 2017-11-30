#include "LogFuncNodeGPUKernel.hpp"

#include <cmath>
#include <assert.h>

LogFuncNodeGPUKernel::LogFuncNodeGPUKernel(const float* const memIn, float* const memOut, size_t size)
    : _memIn(memIn), _memOut(memOut), _size(size)
{
    assert(_memIn != nullptr);
    assert(_memOut != nullptr);
}

LogFuncNodeGPUKernel::~LogFuncNodeGPUKernel() { }

void LogFuncNodeGPUKernel::Run()
{
    for (size_t i = 0; i < _size; ++i)
    {
        _memOut[i] = std::log(_memIn[i]);
    }
}
