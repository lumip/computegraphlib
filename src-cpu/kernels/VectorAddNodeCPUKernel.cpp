#include "VectorAddNodeCPUKernel.hpp"

VectorAddNodeCPUKernel::VectorAddNodeCPUKernel(const float* const memoryA, const float* const memoryB, float* const memoryResult, MemoryDimensions dimA, MemoryDimensions dimB)
    : _memoryA(memoryA), _memoryB(memoryB), _memoryResult(memoryResult), _dimA(dimA), _dimB(dimB)
{ }

VectorAddNodeCPUKernel::~VectorAddNodeCPUKernel() { }

void VectorAddNodeCPUKernel::Run()
{
    for (size_t i = 0; i < _dimA.yDim; ++i)
    {
        for (size_t j = 0; j < _dimA.xDim; ++j)
        {
            size_t globalId = GetIndex(i, j, _dimA.xDim);
            size_t summandId = GetIndex(i % _dimB.yDim, j % _dimB.xDim, _dimB.xDim);
            _memoryResult[globalId] = _memoryA[globalId] + _memoryB[summandId];
            //_memoryResult[i] = _memoryA[i] + _memoryB[i];
        }
    }
}
