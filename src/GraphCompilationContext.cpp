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
    const size_t size = n * dimensions;
    NodeMemoryDescriptor memDesc {this->AllocateMemory(size), n, dimensions, size};
    this->_memoryDescriptors.emplace(memDesc.handle, memDesc);
    return memDesc.handle;
}

void GraphCompilationContext::AssignNodeMemory(const ConstNodePtr node, const NodeMemoryHandle memoryHandle)
{
    this->_memoryMap[node] = memoryHandle;
}

GraphCompilationContext::NodeMemoryDescriptor GraphCompilationContext::GetNodeMemoryDescriptor(const ConstNodePtr node) const
{
    return this->_memoryDescriptors.at(this->_memoryMap.at(node));
}

InputDataBuffer& GraphCompilationContext::GetInputDataBuffer(std::string inputName) const
{
    return this->_inputData.at(inputName);
}
