#include "GraphCompilationContext.hpp"

GraphCompilationContext::GraphCompilationContext(const InputDataMap& inputData)
    : _inputData(inputData)
    , _memoryMap()
{
}

GraphCompilationContext::~GraphCompilationContext() { }

void GraphCompilationContext::RegisterNodeMemory(Node::const_ptr const node, size_t n, size_t dimensions)
{
    if (this->_memoryMap.find(node) != this->_memoryMap.end())
    {
        throw std::invalid_argument("Memory for this node has already been registered.");
    }
    NodeMemory mem;
    mem.n = n;
    mem.dimensions = dimensions;
    mem.size = mem.n * mem.dimensions;
    mem.handle = this->AllocateMemory(mem.size);
    this->_memoryMap[node] = mem;
}

GraphCompilationContext::NodeMemory GraphCompilationContext::GetNodeMemory(Node::const_ptr const node) const
{
    return this->_memoryMap.at(node);
}

InputDataBuffer GraphCompilationContext::GetInputDataBuffer(std::string inputName) const
{
    return this->_inputData.at(inputName);
}
