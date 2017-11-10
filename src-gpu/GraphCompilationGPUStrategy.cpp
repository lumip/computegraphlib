#ifdef GPU

#include "GraphCompilationGPUStrategy.hpp"

GraphCompilationGPUStrategy::GraphCompilationGPUStrategy()
{
}

GraphCompilationGPUStrategy::~GraphCompilationGPUStrategy() { }

NodeMemoryHandle GraphCompilationGPUStrategy::AllocateMemory(size_t size)
{
    throw std::logic_error("not implemented yet.");
}

void GraphCompilationGPUStrategy::DeallocateMemory(const NodeMemoryHandle mem)
{
    throw std::logic_error("not implemented yet.");
}

void GraphCompilationGPUStrategy::EnqueueKernel(std::unique_ptr<const Kernel>&& kernel)
{
    throw std::logic_error("not implemented yet.");
}

void GraphCompilationGPUStrategy::CopyOutputData(const NodeMemoryHandle outputNodeMemory, DataBuffer& outputBuffer, size_t size) const
{
    throw std::logic_error("not implemented yet.");
}

void GraphCompilationGPUStrategy::CopyInputData(const NodeMemoryHandle inputNodeMemory, InputDataBuffer& inputBuffer, size_t size) const
{
    throw std::logic_error("not implemented yet.");
}

void GraphCompilationGPUStrategy::Evaluate() const
{
    throw std::logic_error("not implemented yet.");
}

#endif
