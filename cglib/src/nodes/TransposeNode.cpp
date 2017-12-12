#include "nodes/TransposeNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

TransposeNode::TransposeNode(NodePtr input)
    : Node(false, false)
    , _input(input)
{
    this->SubscribeTo(input);
}

TransposeNode::~TransposeNode() { }

Node::ConstNodeList TransposeNode::GetInputs() const
{
    return ConstNodeList({ _input });
}

std::string TransposeNode::ToString() const
{
    std::stringstream ss;
    ss << "<TransposeNode " << (this) << ">";
    return ss.str();
}

void TransposeNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileTransposeNode(this);
}

void TransposeNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    MemoryDimensions dim = memoryMap.GetNodeMemoryDimensions(_input);
    std::swap(dim.yDim, dim.xDim);
    memoryMap.RegisterNodeMemory(this, MemoryDimensions(dim));
}
