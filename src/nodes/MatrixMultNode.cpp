#include <sstream>

#include "nodes/MatrixMultNode.hpp"

MatrixMultNode::MatrixMultNode(NodePtr a, NodePtr b)
    : _a(a)
    , _b(b)
{

}

MatrixMultNode::~MatrixMultNode() { }

Node::ConstNodeList MatrixMultNode::GetInputs() const
{
    return ConstNodeList({ _a, _b });
}

std::string MatrixMultNode::ToString() const
{
    std::stringstream ss;
    ss << "<MatrixMultNode " << (this) << ">";
    return ss.str();
}

bool MatrixMultNode::IsInitialized() const
{
    return false;
}
