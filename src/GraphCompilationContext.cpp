#include "GraphCompilationContext.hpp"

GraphCompilationContext::GraphCompilationContext(const InputDataMap& inputData)
    : _inputData(inputData)
    , _memoryDescriptors()
    , _memoryMap()
{
}

GraphCompilationContext::~GraphCompilationContext()
{
    this->DeallocateAllMemory();
}

GraphCompilationContext::NodeMemoryHandle GraphCompilationContext::RegisterMemory(size_t n, size_t dimensions)
{
    NodeMemoryDescriptor memDesc;
    memDesc.n = n;
    memDesc.dimensions = dimensions;
    memDesc.size = memDesc.n * memDesc.dimensions;
    memDesc.handle = this->AllocateMemory(memDesc.size);
    this->_memoryDescriptors[memDesc.handle] = memDesc;
    return memDesc.handle;
}

void GraphCompilationContext::AssignNodeMemory(Node::const_ptr const node, const NodeMemoryHandle memoryHandle)
{
    this->_memoryMap[node] = memoryHandle;
}

GraphCompilationContext::NodeMemoryDescriptor GraphCompilationContext::GetNodeMemoryDescriptor(Node::const_ptr const node) const
{
    return this->_memoryDescriptors.at(this->_memoryMap.at(node));
}

InputDataBuffer GraphCompilationContext::GetInputDataBuffer(std::string inputName) const
{
    return this->_inputData.at(inputName);
}
