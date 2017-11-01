#include "nodes/InputNode.hpp"

InputNode::InputNode(size_t dim)
    : _dim(dim)
{
}

InputNode::~InputNode() { }

Node::ConstNodeMap InputNode::GetInputs() const
{
    ConstNodeMap m;
    return m;
}
