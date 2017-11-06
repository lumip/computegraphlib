#include <sstream>

#include "nodes/MatrixMultNode.hpp"

MatrixMultNode::MatrixMultNode(NodePtr a, NodePtr b)
    : _a(a)
    , _b(b)
{

}

MatrixMultNode::~MatrixMultNode() { }

Node::ConstNodeMap MatrixMultNode::GetInputs() const
{
    Node::ConstNodeMap m;
    m.emplace("A", _a);
    m.emplace("B", _b);
    return m;
}

std::string MatrixMultNode::ToString() const
{
    std::stringstream ss;
    ss << "<MatrixMultNode " << (this) << ">";
    return ss.str();
}
