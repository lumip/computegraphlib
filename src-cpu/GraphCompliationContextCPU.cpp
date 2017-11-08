#ifdef CPU
#include "GraphCompilationContext.hpp"

struct CPUContext
{
    std::vector<std::unique_ptr<const Kernel>> kernels;
};

void* GraphCompilationContext::InitContext()
{
    return new CPUContext;
}

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
    delete reinterpret_cast<CPUContext* const>(this->_context);
}

void GraphCompilationContext::EnqueueKernel(std::unique_ptr<const Kernel>& kernel)
{
    reinterpret_cast<CPUContext* const>(this->_context)->kernels.emplace_back(std::move(kernel));
}

#endif
