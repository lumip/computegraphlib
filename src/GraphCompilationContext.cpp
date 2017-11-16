#include "GraphCompilationContext.hpp"

GraphCompilationContext::GraphCompilationContext(const InputDimensionsMap& inputDimensions, std::unique_ptr<GraphCompilationTargetStrategy>&& strategy)
    : _inputDimensions(inputDimensions)
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

NodeMemoryDescriptor GraphCompilationContext::AllocateMemory(const MemoryDimensions memoryDimensions)
{
    NodeMemoryDescriptor memDesc {_strategy->AllocateMemory(memoryDimensions.size()), memoryDimensions};
    this->_memoryDescriptors.emplace(memDesc.handle, memDesc);
    return memDesc;
}

void GraphCompilationContext::AssignNodeMemory(const ConstNodePtr node, const NodeMemoryHandle memoryHandle)
{
    this->_memoryMap.emplace(node, memoryHandle);
}

NodeMemoryDescriptor GraphCompilationContext::GetNodeMemoryDescriptor(const ConstNodePtr node) const
{
    return this->_memoryDescriptors.at(this->_memoryMap.at(node));
}

MemoryDimensions GraphCompilationContext::GetInputDimensions(std::string inputName) const
{
    return this->_inputDimensions.at(inputName);
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
    outputBuffer.resize(memDesc.dimensions.size());
    this->_strategy->CopyOutputData(memHandle, outputBuffer, memDesc.dimensions.size());
}

void GraphCompilationContext::GetNodeData(const ConstNodePtr node, DataBuffer& outputBuffer) const
{
    const NodeMemoryHandle memHandle = this->_memoryMap.at(node);
    const NodeMemoryDescriptor memDesc = this->_memoryDescriptors.at(memHandle);
    outputBuffer.resize(memDesc.dimensions.size());
    this->_strategy->CopyOutputData(memHandle, outputBuffer, memDesc.dimensions.size());
}

void GraphCompilationContext::SetNodeData(const ConstNodePtr node, InputDataBuffer& inputBuffer)
{
    const NodeMemoryHandle memHandle = this->_memoryMap.at(node);
    const NodeMemoryDescriptor memDesc = this->_memoryDescriptors.at(memHandle);
    this->_strategy->CopyInputData(memHandle, inputBuffer, memDesc.dimensions.size());
}

void GraphCompilationContext::EnqueueKernel(std::unique_ptr<Kernel>&& kernel)
{
    _strategy->EnqueueKernel(std::move(kernel));
}

void GraphCompilationContext::Evaluate(const InputDataMap& inputData)
{
    std::vector<std::pair<const NodeMemoryDescriptor, const InputDataBuffer&>> inputs;
    for (auto input : inputData)
    {
        std::string inputName = input.first;
        const InputDataBuffer& inputBuffer = input.second;
        const NodeMemoryHandle memHandle = this->_inputMemoryMap.at(inputName);
        const NodeMemoryDescriptor memDesc = this->_memoryDescriptors.at(memHandle);
        inputs.emplace_back(memDesc, inputBuffer);
    }
    _strategy->Evaluate(inputs);
}
