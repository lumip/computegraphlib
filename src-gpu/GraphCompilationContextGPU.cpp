#ifdef GPU
#include "GraphCompilationContext.hpp"

void* GraphCompilationContext::InitContext()
{
    return nullptr;
}

GraphCompilationContext::NodeMemoryHandle GraphCompilationContext::AllocateMemory(size_t size)
{
    throw std::logic_error("not implemented yet.");
}

void GraphCompilationContext::DeallocateAllMemory()
{
    throw std::logic_error("not implemented yet.");
}

void GraphCompilationContext::EnqueueKernel(std::unique_ptr<const Kernel>&& kernel)
{
    throw std::logic_error("not implemented yet.");
}

void GraphCompilationContext::Evaluate() const
{
    throw std::logic_error("not implemented yet.");
}

void GraphCompilationContext::CopyOutputData(const NodeMemoryHandle outputNodeMemory, DataBuffer& outputBuffer) const
{
    throw std::logic_error("not implemented yet.");
}

#endif
