#ifndef _GRAPH_COMPILATION_CONTEXT_HPP_
#define _GRAPH_COMPILATION_CONTEXT_HPP_

#include <memory>

#include "types.hpp"
#include "Kernel.hpp"
#include "CompiledGraph.hpp"

typedef uint64_t NodeMemoryHandle;
struct NodeMemoryDescriptor
{
    const NodeMemoryHandle handle;
    const MemoryDimensions dimensions;
};

class GraphCompilationTargetStrategy
{
public:
    GraphCompilationTargetStrategy() { }
    virtual ~GraphCompilationTargetStrategy() { }
    virtual NodeMemoryHandle AllocateMemory(size_t size) = 0;
    virtual void DeallocateMemory(const NodeMemoryHandle mem) = 0;
    virtual void EnqueueKernel(std::unique_ptr<const Kernel>&& kernel) = 0;
    virtual void CopyOutputData(const NodeMemoryHandle outputNodeMemory, DataBuffer& outputBuffer, size_t size) const = 0;
    virtual void CopyInputData(const NodeMemoryHandle inputNodeMemory, InputDataBuffer& inputBuffer, size_t size) const = 0;
    virtual void Evaluate() const = 0;
};

class GraphCompilationContext : public CompiledGraph
{
private:
    const std::unique_ptr<GraphCompilationTargetStrategy> _strategy;
    std::map<ConstNodePtr, NodeMemoryHandle> _memoryMap;
    std::map<NodeMemoryHandle, NodeMemoryDescriptor> _memoryDescriptors;
    const InputDimensionsMap _inputDimensions;
    std::map<std::string, const NodeMemoryHandle> _inputMemoryMap;
    std::map<std::string, const NodeMemoryHandle> _outputMemoryMap;
public:
    GraphCompilationContext(const InputDimensionsMap& inputDimensions, std::unique_ptr<GraphCompilationTargetStrategy>&& strategy);
    virtual ~GraphCompilationContext();
    NodeMemoryDescriptor RegisterMemory(const MemoryDimensions memoryDimensions);
    void AssignNodeMemory(const ConstNodePtr node, const NodeMemoryHandle memoryHandle);
    NodeMemoryDescriptor GetNodeMemoryDescriptor(const ConstNodePtr node) const;
    MemoryDimensions GetInputDimensions(std::string inputName) const;
    void RegisterInputMemory(const std::string inputName, const NodeMemoryHandle memoryHandle);
    void RegisterOutputMemory(const std::string outputName, const NodeMemoryHandle memoryHandle);
    void EnqueueKernel(std::unique_ptr<const Kernel>&& kernel);
    void Evaluate() const;
    void GetOutputData(std::string outputName, DataBuffer& outputBuffer) const;
    void GetNodeData(const ConstNodePtr node, DataBuffer& outputBuffer) const;
    void SetNodeData(const ConstNodePtr node, InputDataBuffer& inputBuffer) const;
};

#endif
