#ifndef _GRAPH_COMPILATION_CONTEXT_HPP_
#define _GRAPH_COMPILATION_CONTEXT_HPP_

#include "nodes/Node.hpp"

class GraphCompilationContext
{
public:
    typedef size_t NodeMemoryHandle;
    struct NodeMemory
    {
        GraphCompilationContext::NodeMemoryHandle handle;
        size_t n;
        size_t dimensions;
        size_t size;
    };
private:
    std::map<Node::const_ptr, NodeMemory> _memory_map;
public:
    GraphCompilationContext();
    virtual ~GraphCompilationContext();
    void RegisterNodeMemory(Node::const_ptr const node, size_t n, size_t dimensions);
    NodeMemory GetNodeMemory(Node::const_ptr const node) const;
private:
    NodeMemoryHandle AllocateMemory(size_t size);
};

#endif
