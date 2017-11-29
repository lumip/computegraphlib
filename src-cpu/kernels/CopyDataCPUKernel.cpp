#include "CopyDataCPUKernel.hpp"

CopyDataCPUKernel::CopyDataCPUKernel(const float* const memIn, float* const memOut, size_t count, size_t strideIn, size_t strideOut)
    : _memIn(memIn), _memOut(memOut), _count(count), _strideIn(strideIn), _strideOut(strideOut)
{ }

CopyDataCPUKernel::~CopyDataCPUKernel() { }

void CopyDataCPUKernel::Run()
{
    for (size_t i = 0; i < _count; ++i)
    {
        _memOut[i * _strideOut] = _memIn[i * _strideIn];
    }
}
