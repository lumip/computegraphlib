#include "nodes/ConstMultNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

ConstMultNode::ConstMultNode(NodePtr input, float constantFactor)
    : Node(true, false)
    , _input(input)
    , _factor(constantFactor)
{
    this->SubscribeTo(input);
}

ConstMultNode::~ConstMultNode() { }

Node::ConstNodeList ConstMultNode::GetInputs() const
{
    return ConstNodeList({ _input });
}

std::string ConstMultNode::ToString() const
{
    std::stringstream ss;
    ss << "<ConstMultNode " << (this) << ">";
    return ss.str();
}

void ConstMultNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileConstMultNode(this);
}

void ConstMultNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    const MemoryDimensions dim = memoryMap.GetNodeMemoryDimensions(_input);
    memoryMap.RegisterNodeMemory(this, MemoryDimensions(dim));
}

float ConstMultNode::GetFactor() const
{
    return _factor;
}
