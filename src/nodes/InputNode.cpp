#include "nodes/InputNode.hpp"

InputNode::InputNode(std::string name, size_t dim)
    : _name(name)
    , _dim(dim)
{
}

InputNode::~InputNode() { }

Node::ConstNodeList InputNode::GetInputs() const
{
    return ConstNodeList();
}

std::string InputNode::ToString() const
{
    return "<InputNode " + _name + ">";
}

bool InputNode::IsInitialized() const
{
    return true;
}
