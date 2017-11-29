#include "VectorAddNodeCPUKernel.hpp"

VectorAddNodeCPUKernel::VectorAddNodeCPUKernel(const float* const memoryA, const float* const memoryB, float* const memoryResult, size_t size)
    : _memoryA(memoryA), _memoryB(memoryB), _memoryResult(memoryResult), _size(size)
{ }

VectorAddNodeCPUKernel::~VectorAddNodeCPUKernel() { }

void VectorAddNodeCPUKernel::Run()
{
    for (size_t i = 0; i < _size; ++i)
    {
        _memoryResult[i] = _memoryA[i] + _memoryB[i];
    }
}
