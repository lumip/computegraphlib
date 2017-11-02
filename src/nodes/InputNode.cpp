#include "nodes/InputNode.hpp"

InputNode::InputNode(std::string name, size_t dim)
    : _name(name)
    , _dim(dim)
{
}

InputNode::~InputNode() { }

Node::ConstNodeMap InputNode::GetInputs() const
{
    ConstNodeMap m;
    return m;
}
