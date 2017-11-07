#include "nodes/VectorAddNode.hpp"
#include <sstream>

VectorAddNode::VectorAddNode(NodePtr a, NodePtr b)
    : Node(), 
    _summandA(a), 
    _summandB(b)
{
    this->SubscribeTo(_summandA);
    this->SubscribeTo(_summandB);
}

VectorAddNode::~VectorAddNode() { }

Node::ConstNodeList VectorAddNode::GetInputs() const
{
    return ConstNodeList({ _summandA, _summandB });
}

std::string VectorAddNode::ToString() const
{
    std::stringstream ss;
    ss << "<VectorAddNode " << (this) << ">";
    return ss.str();
}

bool VectorAddNode::IsInitialized() const
{
    return false;
}
