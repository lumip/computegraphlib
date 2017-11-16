#include "nodes/VariableNode.hpp"

VariableNode::VariableNode(std::string name, size_t xDim, size_t yDim)
    : _name(name)
    , _xDim(xDim)
    , _yDim(yDim)
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

bool VariableNode::IsInitialized() const
{
    return true;
}

void VariableNode::Compile(GraphCompilationContext& context, NodeCompiler& nodeCompiler) const
{
    throw new std::logic_error("not yet implemented.");
}

MemoryDimensions VariableNode::GetMemoryDimensions(const InputDimensionsMap& inputDimensions, const std::map<ConstNodePtr, MemoryDimensions>& nodeMemoryDimensions) const
{
    throw new std::logic_error("not yet implemented.");
}
