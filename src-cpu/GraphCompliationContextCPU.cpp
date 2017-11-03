#ifdef CPU
#include "GraphCompilationContext.hpp"

GraphCompilationContext::NodeMemoryHandle GraphCompilationContext::AllocateMemory(size_t size)
{
    return reinterpret_cast<NodeMemoryHandle>(new float[size]);
}

void GraphCompilationContext::DeallocateAllMemory()
{
    for (auto elem : _memoryDescriptors)
    {
        NodeMemoryHandle buf = reinterpret_cast<NodeMemoryHandle>(elem.first);
        delete[](buf);
    }
}

#endif
