#include "nodes/SliceNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

SliceNode::SliceNode(NodePtr input, size_t slice_id, size_t axis)
    : _input(input)
    , _slice_id(slice_id)
    , _axis(axis)
{
    this->SubscribeTo(input);
}

SliceNode::~SliceNode() { }

Node::ConstNodeList SliceNode::GetInputs() const
{
    return ConstNodeList({ _input });
}

std::string SliceNode::ToString() const
{
    std::stringstream ss;
    ss << "<SliceNode " << (this) << ">";
    return ss.str();
}

bool SliceNode::IsInitialized() const
{
    return false;
}

void SliceNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileSliceNode(this);
}

void SliceNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    MemoryDimensions dim = memoryMap.GetNodeMemoryDimensions(_input);
    dim.dims[_axis] = 1;
    memoryMap.RegisterNodeMemory(this, MemoryDimensions(dim));
}
