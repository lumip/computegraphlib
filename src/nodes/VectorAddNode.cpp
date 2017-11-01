#include "nodes/VectorAddNode.hpp"

VectorAddNode::VectorAddNode(Node::ptr a, Node::ptr b) 
    : Node(), 
    _summandA(a), 
    _summandB(b)
{
}

VectorAddNode::~VectorAddNode() { }

Node::ConstNodeMap VectorAddNode::GetInputs() const
{
    ConstNodeMap m;
    m["SummandA"] = _summandA;
    m["SummandB"] = _summandB;
    return m;
}
