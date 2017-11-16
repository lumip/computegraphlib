#ifndef _GRAPH_COMPILATION_PLATFORM_HPP_
#define _GRAPH_COMPILATION_PLATFORM_HPP_

#include "types.hpp"

typedef uint64_t NodeMemoryHandle;
struct NodeMemoryDescriptor
{
    const NodeMemoryHandle handle;
    const MemoryDimensions dimensions;
};

class GraphCompilationPlatform
{
public:
    GraphCompilationPlatform() { }
    virtual ~GraphCompilationPlatform() { }
    virtual NodeMemoryHandle AllocateMemory(size_t size) = 0;
    virtual void DeallocateMemory(const NodeMemoryHandle mem) = 0;
    virtual void EnqueueKernel(std::unique_ptr<Kernel>&& kernel) = 0;
    virtual void CopyOutputData(const NodeMemoryHandle outputNodeMemory, DataBuffer& outputBuffer, size_t size) const = 0;
    virtual void CopyInputData(const NodeMemoryHandle inputNodeMemory, InputDataBuffer& inputBuffer, size_t size) = 0;
    virtual void Evaluate(const std::vector<std::pair<const NodeMemoryDescriptor, InputDataBuffer&>>& inputData) = 0;
};

#endif
