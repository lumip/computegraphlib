#include "CompilationMemoryMap.hpp"

CompilationMemoryMap::CompilationMemoryMap(const InputDimensionsMap& inputDimensions)
    : _inputDimensions(inputDimensions)
    , _memoryMap()
    , _inputMemoryMap()
    , _outputMemoryMap()
{
}

CompilationMemoryMap::~CompilationMemoryMap() { }

void CompilationMemoryMap::RegisterNodeMemory(const ConstNodePtr node, const MemoryDimensions memoryDimensions)
{
    this->_memoryMap.emplace(node, memoryDimensions);
}

MemoryDimensions CompilationMemoryMap::GetNodeMemoryDimensions(const ConstNodePtr node) const
{
    return this->_memoryMap.at(node);
}

MemoryDimensions CompilationMemoryMap::GetInputDimensions(std::string inputName) const
{
    return this->_inputDimensions.at(inputName);
}

void CompilationMemoryMap::RegisterInputMemory(const std::string inputName, const ConstNodePtr node)
{
    this->_inputMemoryMap.emplace(inputName, node);
}

void CompilationMemoryMap::RegisterOutputMemory(const std::string outputName, const ConstNodePtr node)
{
    this->_outputMemoryMap.emplace(outputName, node);
}

ConstNodePtr CompilationMemoryMap::GetInputNode(const std::string inputName) const
{
    return this->_inputMemoryMap.at(inputName);
}

ConstNodePtr CompilationMemoryMap::GetOutputNode(const std::string outputName) const
{
    return this->_outputMemoryMap.at(outputName);
}
