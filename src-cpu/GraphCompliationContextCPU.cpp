#ifdef CPU
#include "GraphCompilationContext.hpp"

GraphCompilationContext::NodeMemoryHandle GraphCompilationContext::AllocateMemory(size_t size)
{
    return reinterpret_cast<NodeMemoryHandle>(new float[size]);
}
#endif
