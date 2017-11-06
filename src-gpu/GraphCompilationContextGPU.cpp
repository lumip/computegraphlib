#ifdef GPU
#include "GraphCompilationContext.hpp"

GraphCompilationContext::NodeMemoryHandle GraphCompilationContext::AllocateMemory(size_t size)
{
    throw std::logic_error("not implemented yet.");
}

void GraphCompilationContext::DeallocateAllMemory()
{
    throw std::logic_error("not implemented yet.");
}

#endif
