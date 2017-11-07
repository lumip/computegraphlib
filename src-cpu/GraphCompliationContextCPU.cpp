#ifdef CPU
#include "GraphCompilationContext.hpp"

GraphCompilationContext::NodeMemoryHandle GraphCompilationContext::AllocateMemory(size_t size)
{
    return static_cast<NodeMemoryHandle>(new float[size]);
}

void GraphCompilationContext::DeallocateAllMemory()
{
    for (auto elem : _memoryDescriptors)
    {
        NodeMemoryHandle buf = static_cast<NodeMemoryHandle>(elem.first);
        delete[](buf);
    }
}

#endif
