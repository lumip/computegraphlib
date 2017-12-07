#include "CompilationMemoryMap.hpp"

CompilationMemoryMap::CompilationMemoryMap(const InputDimensionsMap inputDimensions)
    : _inputDimensions(std::move(inputDimensions))
    , _memoryMap()
    , _inputMemoryMap()
{
}

CompilationMemoryMap::CompilationMemoryMap() : CompilationMemoryMap(InputDimensionsMap()) { }

CompilationMemoryMap::~CompilationMemoryMap() { }

CompilationMemoryMap::CompilationMemoryMap(const CompilationMemoryMap& other)
    : _inputDimensions(other._inputDimensions)
    , _memoryMap(other._memoryMap)
    , _inputMemoryMap(other._inputMemoryMap)
{
}

CompilationMemoryMap::CompilationMemoryMap(CompilationMemoryMap&& other)
    : _inputDimensions(), _memoryMap(), _inputMemoryMap()
{
    swap(*this, other);
}

void CompilationMemoryMap::swap(CompilationMemoryMap& a, CompilationMemoryMap& b)
{
    std::swap(a._inputDimensions, b._inputDimensions);
    std::swap(a._memoryMap, b._memoryMap);
    std::swap(a._inputMemoryMap, b._inputMemoryMap);
}

CompilationMemoryMap& CompilationMemoryMap::operator=(CompilationMemoryMap other)
{
    swap(*this, other);
    return *this;
}

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

ConstNodePtr CompilationMemoryMap::GetInputNode(const std::string inputName) const
{
    return this->_inputMemoryMap.at(inputName);
}
