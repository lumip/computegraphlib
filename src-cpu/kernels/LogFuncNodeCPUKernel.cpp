#include "LogFuncNodeCPUKernel.hpp"

#include <cmath>

LogFuncNodeCPUKernel::LogFuncNodeCPUKernel(const float* const memIn, float* const memOut, size_t size)
    : _memIn(memIn), _memOut(memOut), _size(size)
{ }

LogFuncNodeCPUKernel::~LogFuncNodeCPUKernel() { }

void LogFuncNodeCPUKernel::Run()
{
    for (size_t i = 0; i < _size; ++i)
    {
        _memOut[i] = std::log(_memIn[i]);
    }
}
