#include "VectorMultNodeCPUKernel.hpp"

#include <assert.h>

VectorMultNodeCPUKernel::VectorMultNodeCPUKernel(const float* const memA, const float* const memB, float* const memRes, MemoryDimensions dimA, MemoryDimensions dimB)
    : _memA(memA), _memB(memB), _memRes(memRes), _dimA(dimA), _dimB(dimB)
{
    assert(_memA != nullptr);
    assert(_memB != nullptr);
    assert(_memRes != nullptr);
    assert(((_dimA.xDim == _dimB.xDim) && ((_dimA.yDim % _dimB.yDim == 0) || (_dimB.yDim % _dimA.yDim == 0))) ||
           ((_dimA.yDim == _dimB.yDim) && ((_dimA.xDim % _dimB.xDim == 0) || (_dimB.xDim % _dimA.xDim == 0))));
}

VectorMultNodeCPUKernel::~VectorMultNodeCPUKernel() { }

void VectorMultNodeCPUKernel::Run()
{
    for (size_t i = 0; i < _dimA.yDim; ++i)
    {
        for (size_t j = 0; j < _dimA.xDim; ++j)
        {
            size_t globalId = GetIndex(i, j, _dimA.xDim);
            size_t summandId = GetIndex(i % _dimB.yDim, j % _dimB.xDim, _dimB.xDim);
            _memRes[globalId] = _memA[globalId] * _memB[summandId];
        }
    }
}
