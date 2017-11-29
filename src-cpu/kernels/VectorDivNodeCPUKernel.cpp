#include "VectorDivNodeCPUKernel.hpp"

VectorDivNodeCPUKernel::VectorDivNodeCPUKernel(const float* const memA, const float* const memB, float* const memRes, size_t size)
    : _memA(memA), _memB(memB), _memRes(memRes), _size(size)
{ }

VectorDivNodeCPUKernel::~VectorDivNodeCPUKernel() { }

void VectorDivNodeCPUKernel::Run()
{
    for (size_t i = 0; i < _size; ++i)
    {
        _memRes[i] = _memA[i] / _memB[i];
    }
}
