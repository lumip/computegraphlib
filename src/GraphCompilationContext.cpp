#include "GraphCompilationContext.hpp"

GraphCompilationContext::GraphCompilationContext(const InputDataMap& inputData, std::unique_ptr<GraphCompilationTargetStrategy>&& strategy)
    : _inputData(inputData)
    , _strategy(std::move(strategy))
    , _memoryDescriptors()
    , _memoryMap()
    , _inputMemoryMap()
    , _outputMemoryMap()
{
}

GraphCompilationContext::~GraphCompilationContext()
{
    for (auto elem : _memoryDescriptors)
    {
        _strategy->DeallocateMemory(elem.first);
    }
}

NodeMemoryHandle GraphCompilationContext::RegisterMemory(size_t yDim, size_t xDim)
{
    const size_t size = yDim * xDim;
    NodeMemoryDescriptor memDesc {_strategy->AllocateMemory(size), yDim, xDim};
    this->_memoryDescriptors.emplace(memDesc.handle, memDesc);
    return memDesc.handle;
}

void GraphCompilationContext::AssignNodeMemory(const ConstNodePtr node, const NodeMemoryHandle memoryHandle)
{
    this->_memoryMap.emplace(node, memoryHandle);
}

NodeMemoryDescriptor GraphCompilationContext::GetNodeMemoryDescriptor(const ConstNodePtr node) const
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
    const NodeMemoryDescriptor memDesc = this->_memoryDescriptors.at(memHandle);
    outputBuffer.resize(memDesc.size());
    this->_strategy->CopyOutputData(memHandle, outputBuffer, memDesc.size());
}

void GraphCompilationContext::EnqueueKernel(std::unique_ptr<const Kernel>&& kernel)
{
    _strategy->EnqueueKernel(std::move(kernel));
}

void GraphCompilationContext::Evaluate() const
{
    _strategy->Evaluate();
}
