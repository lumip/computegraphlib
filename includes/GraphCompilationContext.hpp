#ifndef _GRAPH_COMPILATION_CONTEXT_HPP_
#define _GRAPH_COMPILATION_CONTEXT_HPP_

#include "types.hpp"
#include "nodes/Node.hpp"

class GraphCompilationContext
{
public:
    typedef size_t NodeMemoryHandle;
    struct NodeMemory
    {
        GraphCompilationContext::NodeMemoryHandle handle;
        size_t n; // todo: decouple compilation from input data
        size_t dimensions;
        size_t size;
    };
private:
    std::map<Node::const_ptr, NodeMemory> _memoryMap;
    const InputDataMap& _inputData;
public:
    GraphCompilationContext(const InputDataMap& inputData); // todo: decouple compilation from input data
    virtual ~GraphCompilationContext();
    void RegisterNodeMemory(Node::const_ptr const node, size_t n, size_t dimensions);
    NodeMemory GetNodeMemory(Node::const_ptr const node) const;
    InputDataBuffer GetInputDataBuffer(std::string inputName) const;
private:
    NodeMemoryHandle AllocateMemory(size_t size);
};

#endif
