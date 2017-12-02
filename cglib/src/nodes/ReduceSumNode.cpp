#include "nodes/ReduceSumNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

ReduceSumNode::ReduceSumNode(NodePtr input, size_t axis)
    : _input(input)
    , _axis(axis)
{
    this->SubscribeTo(input);
}

ReduceSumNode::~ReduceSumNode() { }

Node::ConstNodeList ReduceSumNode::GetInputs() const
{
    return ConstNodeList({ _input });
}

std::string ReduceSumNode::ToString() const
{
    std::stringstream ss;
    ss << "<ReduceSumNode " << (this) << ">";
    return ss.str();
}

bool ReduceSumNode::IsInitialized() const
{
    return false;
}

void ReduceSumNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileReduceSumNode(this);
}

void ReduceSumNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    MemoryDimensions dim = memoryMap.GetNodeMemoryDimensions(_input);
    dim.dims[_axis] = 1;
    memoryMap.RegisterNodeMemory(this, MemoryDimensions(dim));
}

size_t ReduceSumNode::GetAxis() const
{
    return _axis;
}
