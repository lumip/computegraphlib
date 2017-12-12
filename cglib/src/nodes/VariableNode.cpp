#include "nodes/VariableNode.hpp"

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

VariableNode::VariableNode(std::string name)
    : Node(true, true)
    , _name(name)
    , _input(nullptr)
{
}

VariableNode::~VariableNode() {}

void VariableNode::SetInput(const NodePtr inputNode)
{
    if (_input != nullptr)
    {
        this->UnsubscribeFrom(_input);
    }
    _input = inputNode;
    if (_input != nullptr)
    {
        this->SubscribeTo(_input);
    }
}

Node::ConstNodeList VariableNode::GetInputs() const
{
    if (_input != nullptr)
    {
        return ConstNodeList({ _input });
    }
    return ConstNodeList{};
}

std::string VariableNode::ToString() const
{
    return "<VariableNode " + _name + ">";
}

void VariableNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileVariableNode(this);
}

void VariableNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    const MemoryDimensions initDim = memoryMap.GetInputDimensions(_name);
    memoryMap.RegisterNodeMemory(this, initDim);
    memoryMap.RegisterVariableNode(_name, this);
}
