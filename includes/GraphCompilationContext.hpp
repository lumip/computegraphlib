#ifndef _GRAPH_COMPILATION_CONTEXT_HPP_
#define _GRAPH_COMPILATION_CONTEXT_HPP_

#include "types.hpp"
#include "nodes/Node.hpp"

class GraphCompilationContext
{
public:
    typedef float* NodeMemoryHandle;
    struct NodeMemoryDescriptor
    {
        GraphCompilationContext::NodeMemoryHandle handle;
        size_t n; // todo: decouple compilation from input data
        size_t dimensions;
        size_t size;
    };
private:
    std::map<Node::const_ptr, NodeMemoryHandle> _memoryMap;
    std::map<NodeMemoryHandle, NodeMemoryDescriptor> _memoryDescriptors;
    const InputDataMap& _inputData;
public:
    GraphCompilationContext(const InputDataMap& inputData); // todo: decouple compilation from input data
    virtual ~GraphCompilationContext();
    NodeMemoryHandle RegisterMemory(size_t n, size_t dimensions);
    void AssignNodeMemory(Node::const_ptr const node, const NodeMemoryHandle memoryHandle);
    NodeMemoryDescriptor GetNodeMemoryDescriptor(Node::const_ptr const node) const;
    InputDataBuffer GetInputDataBuffer(std::string inputName) const;
private:
    NodeMemoryHandle AllocateMemory(size_t size);
    void DeallocateAllMemory();
};

#endif
