#include "nodes/VariableNode.hpp"

VariableNode::VariableNode(std::string name, size_t dim)
    : _name(name)
    , _dim(dim)
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

Node::ConstNodeMap VariableNode::GetInputs() const
{
    ConstNodeMap m;
    if (_input != nullptr)
    {
        m[_name] = _input;
    }
    return m;
}

std::string VariableNode::ToString() const
{
    return "<VariableNode " + _name + ">";
}
