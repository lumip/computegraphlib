#include "nodes/VectorAddNode.hpp"

VectorAddNode::VectorAddNode(Node::ptr a, Node::ptr b) 
    : Node(), 
    _summandA(a), 
    _summandB(b)
{
    this->SubscribeTo(_summandA);
    this->SubscribeTo(_summandB);
}

VectorAddNode::~VectorAddNode() { }

Node::ConstNodeMap VectorAddNode::GetInputs() const
{
    ConstNodeMap m;
    m["SummandA"] = _summandA;
    m["SummandB"] = _summandB;
    return m;
}

std::string VectorAddNode::ToString() const
{
    return "<VectorAddNode " + _summandA->ToString() + " + " + _summandB->ToString() + ">";
}
