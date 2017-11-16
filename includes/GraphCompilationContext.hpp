#ifndef _GRAPH_COMPILATION_CONTEXT_HPP_
#define _GRAPH_COMPILATION_CONTEXT_HPP_

#include <memory>

#include "types.hpp"
#include "Kernel.hpp"
#include "CompiledGraph.hpp"
#include "GraphCompilationPlatform.hpp"

class GraphCompilationContext : public CompiledGraph
{
private:
    const std::unique_ptr<GraphCompilationPlatform> _strategy;
    std::map<ConstNodePtr, NodeMemoryHandle> _memoryMap;
    std::map<NodeMemoryHandle, NodeMemoryDescriptor> _memoryDescriptors;
    const InputDimensionsMap _inputDimensions;
    std::map<std::string, const NodeMemoryHandle> _inputMemoryMap;
    std::map<std::string, const NodeMemoryHandle> _outputMemoryMap;
public:
    GraphCompilationContext(const InputDimensionsMap& inputDimensions, std::unique_ptr<GraphCompilationPlatform>&& strategy);
    virtual ~GraphCompilationContext();
    NodeMemoryDescriptor AllocateMemory(const MemoryDimensions memoryDimensions);
    void AssignNodeMemory(const ConstNodePtr node, const NodeMemoryHandle memoryHandle);
    NodeMemoryDescriptor GetNodeMemoryDescriptor(const ConstNodePtr node) const;
    MemoryDimensions GetInputDimensions(std::string inputName) const;
    void RegisterInputMemory(const std::string inputName, const NodeMemoryHandle memoryHandle);
    void RegisterOutputMemory(const std::string outputName, const NodeMemoryHandle memoryHandle);
    void EnqueueKernel(std::unique_ptr<Kernel>&& kernel);
    void Evaluate(const InputDataMap& inputData);
    void GetOutputData(std::string outputName, DataBuffer& outputBuffer) const;
    void GetNodeData(const ConstNodePtr node, DataBuffer& outputBuffer) const;
    void SetNodeData(const ConstNodePtr node, InputDataBuffer& inputBuffer);
};

#endif
