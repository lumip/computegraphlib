#include "GraphCompilationContext.hpp"

GraphCompilationContext::GraphCompilationContext()
    : _memory_map()
{
}

GraphCompilationContext::~GraphCompilationContext() { }

void GraphCompilationContext::RegisterNodeMemory(Node::const_ptr const node, size_t n, size_t dimensions)
{
    NodeMemory mem;
    mem.n = n;
    mem.dimensions = dimensions;
    mem.size = mem.n * mem.dimensions;
    mem.handle = this->AllocateMemory(mem.size);
    this->_memory_map[node] = mem;
}

GraphCompilationContext::NodeMemory GraphCompilationContext::GetNodeMemory(Node::const_ptr const node) const
{
    return this->_memory_map.at(node);
}
