#include "nodes/ReduceMeanNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

ReduceMeanNode::ReduceMeanNode(NodePtr input, size_t axis)
    : _input(input)
    , _axis(axis)
{
    this->SubscribeTo(input);
}

ReduceMeanNode::~ReduceMeanNode() { }

Node::ConstNodeList ReduceMeanNode::GetInputs() const
{
    return ConstNodeList({ _input });
}

std::string ReduceMeanNode::ToString() const
{
    std::stringstream ss;
    ss << "<ReduceMeanNode " << (this) << ">";
    return ss.str();
}

bool ReduceMeanNode::IsInitialized() const
{
    return false;
}

void ReduceMeanNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileReduceMeanNode(this);
}

void ReduceMeanNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    MemoryDimensions dim = memoryMap.GetNodeMemoryDimensions(_input);
    dim.dims[_axis] = 1;
    memoryMap.RegisterNodeMemory(this, MemoryDimensions(dim));
}
