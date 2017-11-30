#include "LogFuncNodeCPUKernel.hpp"

#include <cmath>
#include <assert.h>

LogFuncNodeCPUKernel::LogFuncNodeCPUKernel(const float* const memIn, float* const memOut, size_t size)
    : _memIn(memIn), _memOut(memOut), _size(size)
{
    assert(_memIn != nullptr);
    assert(_memOut != nullptr);
}

LogFuncNodeCPUKernel::~LogFuncNodeCPUKernel() { }

void LogFuncNodeCPUKernel::Run()
{
    for (size_t i = 0; i < _size; ++i)
    {
        _memOut[i] = std::log(_memIn[i]);
    }
}
