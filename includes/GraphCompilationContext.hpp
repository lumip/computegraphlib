#ifndef _GRAPH_COMPILATION_CONTEXT_HPP_
#define _GRAPH_COMPILATION_CONTEXT_HPP_

#include "types.hpp"

class GraphCompilationContext
{
public:
    typedef float* NodeMemoryHandle;
    struct NodeMemoryDescriptor
    {
        const GraphCompilationContext::NodeMemoryHandle handle;
        size_t n; // todo: decouple compilation from input data
        size_t dimensions;
        size_t size;
    };
private:
    std::map<ConstNodePtr, NodeMemoryHandle> _memoryMap;
    std::map<NodeMemoryHandle, NodeMemoryDescriptor> _memoryDescriptors;
    const InputDataMap& _inputData;
public:
    GraphCompilationContext(const InputDataMap& inputData); // todo: decouple compilation from input data
    virtual ~GraphCompilationContext();
    NodeMemoryHandle RegisterMemory(size_t n, size_t dimensions);
    void AssignNodeMemory(const ConstNodePtr node, const NodeMemoryHandle memoryHandle);
    NodeMemoryDescriptor GetNodeMemoryDescriptor(const ConstNodePtr node) const;
    InputDataBuffer& GetInputDataBuffer(std::string inputName) const;
private:
    NodeMemoryHandle AllocateMemory(size_t size);
    void DeallocateAllMemory();
};

#endif
