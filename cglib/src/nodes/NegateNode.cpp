#include "nodes/NegateNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

NegateNode::NegateNode(NodePtr input)
    : _input(input)
{
    this->SubscribeTo(input);
}

NegateNode::~NegateNode() { }

Node::ConstNodeList NegateNode::GetInputs() const
{
    return ConstNodeList({ _input });
}

std::string NegateNode::ToString() const
{
    std::stringstream ss;
    ss << "<NegateNode " << (this) << ">";
    return ss.str();
}

bool NegateNode::IsInitialized() const
{
    return false;
}

void NegateNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileNegateNode(this);
}

void NegateNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    const MemoryDimensions dim = memoryMap.GetNodeMemoryDimensions(_input);
    memoryMap.RegisterNodeMemory(this, MemoryDimensions(dim));
}
