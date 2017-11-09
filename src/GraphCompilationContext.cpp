#include "GraphCompilationContext.hpp"

GraphCompilationContext::GraphCompilationContext(const InputDataMap& inputData)
    : _inputData(inputData)
    , _context(InitContext())
    , _memoryDescriptors()
    , _memoryMap()
    , _inputMemoryMap()
    , _outputMemoryMap()
{
}

GraphCompilationContext::~GraphCompilationContext()
{
    this->DeallocateAllMemory();
}

GraphCompilationContext::NodeMemoryHandle GraphCompilationContext::RegisterMemory(size_t yDim, size_t xDim)
{
    const size_t size = yDim * xDim;
    NodeMemoryDescriptor memDesc {this->AllocateMemory(size), yDim, xDim, size};
    this->_memoryDescriptors.emplace(memDesc.handle, memDesc);
    return memDesc.handle;
}

void GraphCompilationContext::AssignNodeMemory(const ConstNodePtr node, const NodeMemoryHandle memoryHandle)
{
    this->_memoryMap.emplace(node, memoryHandle);
}

GraphCompilationContext::NodeMemoryDescriptor GraphCompilationContext::GetNodeMemoryDescriptor(const ConstNodePtr node) const
{
    return this->_memoryDescriptors.at(this->_memoryMap.at(node));
}

InputDataBuffer& GraphCompilationContext::GetInputDataBuffer(std::string inputName) const
{
    return this->_inputData.at(inputName);
}

void GraphCompilationContext::RegisterInputMemory(const std::string inputName, const NodeMemoryHandle memoryHandle)
{
    this->_inputMemoryMap.emplace(inputName, memoryHandle);
}

void GraphCompilationContext::RegisterOutputMemory(const std::string outputName, const NodeMemoryHandle memoryHandle)
{
    this->_outputMemoryMap.emplace(outputName, memoryHandle);
}

void GraphCompilationContext::GetOutputData(std::string outputName, DataBuffer& outputBuffer) const
{
    const NodeMemoryHandle memHandle = this->_outputMemoryMap.at(outputName);
    this->CopyOutputData(memHandle, outputBuffer);
}
