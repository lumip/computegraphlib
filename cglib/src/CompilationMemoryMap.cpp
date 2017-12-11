#include "CompilationMemoryMap.hpp"

CompilationMemoryMap::CompilationMemoryMap(const InputDimensionsMap inputDimensions)
    : _inputDimensions(std::move(inputDimensions))
    , _memoryDimensions()
    , _inputNodes()
    , _variableNodes()
{
}

CompilationMemoryMap::CompilationMemoryMap() : CompilationMemoryMap(InputDimensionsMap()) { }

CompilationMemoryMap::~CompilationMemoryMap() { }

CompilationMemoryMap::CompilationMemoryMap(const CompilationMemoryMap& other)
    : _inputDimensions(other._inputDimensions)
    , _memoryDimensions(other._memoryDimensions)
    , _inputNodes(other._inputNodes)
    , _variableNodes(other._variableNodes)
{
}

CompilationMemoryMap::CompilationMemoryMap(CompilationMemoryMap&& other)
    : _inputDimensions(), _memoryDimensions(), _inputNodes(), _variableNodes()
{
    swap(*this, other);
}

void CompilationMemoryMap::swap(CompilationMemoryMap& a, CompilationMemoryMap& b)
{
    std::swap(a._inputDimensions, b._inputDimensions);
    std::swap(a._memoryDimensions, b._memoryDimensions);
    std::swap(a._inputNodes, b._inputNodes);
    std::swap(a._variableNodes, b._variableNodes);
}

CompilationMemoryMap& CompilationMemoryMap::operator=(CompilationMemoryMap other)
{
    swap(*this, other);
    return *this;
}

void CompilationMemoryMap::RegisterNodeMemory(const ConstNodePtr node, const MemoryDimensions memoryDimensions)
{
    this->_memoryDimensions.emplace(node, memoryDimensions);
}

MemoryDimensions CompilationMemoryMap::GetNodeMemoryDimensions(const ConstNodePtr node) const
{
    return this->_memoryDimensions.at(node);
}

MemoryDimensions CompilationMemoryMap::GetInputDimensions(const std::string& inputName) const
{
    return this->_inputDimensions.at(inputName);
}

void CompilationMemoryMap::RegisterInputNode(const std::string& inputName, const ConstNodePtr node)
{
    this->_inputNodes.emplace(inputName, node);
}

ConstNodePtr CompilationMemoryMap::GetInputNode(const std::string& inputName) const
{
    return this->_inputNodes.at(inputName);
}

void CompilationMemoryMap::RegisterVariableNode(const std::string& variableName, const ConstNodePtr node)
{
    this->_variableNodes.emplace(variableName, node);
}

ConstNodePtr CompilationMemoryMap::GetVariableNode(const std::string& variableName) const
{
    return this->_variableNodes.at(variableName);
}

std::vector<std::string> CompilationMemoryMap::GetInputNames() const
{
    std::vector<std::string> names;
    for (auto elem : this->_inputNodes)
    {
        names.push_back(elem.first);
    }
    return names;
}

std::vector<std::string> CompilationMemoryMap::GetVariableNames() const
{
    std::vector<std::string> names;
    for (auto elem : this->_variableNodes)
    {
        names.push_back(elem.first);
    }
    return names;
}
