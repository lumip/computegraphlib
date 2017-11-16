#include "GraphCompilationContext.hpp"

MemoryCompilationMap::MemoryCompilationMap(const InputDimensionsMap& inputDimensions)
    : _inputDimensions(inputDimensions)
    , _memoryMap()
    , _inputMemoryMap()
    , _outputMemoryMap()
{
}

MemoryCompilationMap::~MemoryCompilationMap() { }

void MemoryCompilationMap::RegisterNodeMemory(const ConstNodePtr node, const MemoryDimensions memoryDimensions)
{
    this->_memoryMap.emplace(node, memoryDimensions);
}

MemoryDimensions MemoryCompilationMap::GetNodeMemoryDimensions(const ConstNodePtr node) const
{
    return this->_memoryMap.at(node);
}

MemoryDimensions MemoryCompilationMap::GetInputDimensions(std::string inputName) const
{
    return this->_inputDimensions.at(inputName);
}

void MemoryCompilationMap::RegisterInputMemory(const std::string inputName, const ConstNodePtr node)
{
    this->_inputMemoryMap.emplace(inputName, node);
}

void MemoryCompilationMap::RegisterOutputMemory(const std::string outputName, const ConstNodePtr node)
{
    this->_outputMemoryMap.emplace(outputName, node);
}

ConstNodePtr MemoryCompilationMap::GetInputNode(const std::string inputName) const
{
    return this->_inputMemoryMap.at(inputName);
}

ConstNodePtr MemoryCompilationMap::GetOutputNode(const std::string outputName) const
{
    return this->_outputMemoryMap.at(outputName);
}
