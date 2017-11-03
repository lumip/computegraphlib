#ifdef CPU
#include <iostream>
#include "nodes/VectorAddNode.hpp"
#include "GraphCompilationContext.hpp"
#include "Kernel.hpp"

class VectorAddNodeCPUKernel : public Kernel
{
private:
    const float* const _memoryA;
    const float* const _memoryB;
    float* const _memoryResult;
    const size_t _size;
public:
    VectorAddNodeCPUKernel(const float* const memoryA, const float* const memoryB, float* const memoryResult, size_t size)
        : _memoryA(memoryA)
        , _memoryB(memoryB)
        , _memoryResult(memoryResult)
        , _size(size)
    {
    }
    virtual ~VectorAddNodeCPUKernel() { }
    void Run()
    {
        for (size_t i = 0; i < _size; ++i)
        {
            _memoryResult[i] = _memoryA[i] + _memoryB[i];
        }
    }
};

std::unique_ptr<const Kernel> VectorAddNode::Compile(GraphCompilationContext* const context) const
{
    GraphCompilationContext::NodeMemoryDescriptor memDescA = context->GetNodeMemoryDescriptor(this->_summandA);
    GraphCompilationContext::NodeMemoryDescriptor memDescB = context->GetNodeMemoryDescriptor(this->_summandB);
    if (memDescA.size != memDescB.size)
    {
        throw std::runtime_error("Inputs to VectorAddNode have different size.");
    }
    GraphCompilationContext::NodeMemoryHandle mem = context->RegisterMemory(memDescA.n, memDescA.dimensions);
    context->AssignNodeMemory(this, mem);
    return std::unique_ptr<const Kernel>(new VectorAddNodeCPUKernel(memDescA.handle, memDescB.handle, mem, memDescA.size)); // std::make_unique only since c++14
}

#endif
