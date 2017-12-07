#include "nodes/SliceNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

SliceNode::SliceNode(NodePtr input, size_t slice_id, size_t axis)
    : Node(true, false)
    , _input(input)
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

size_t SliceNode::GetSliceId() const
{
    return _slice_id;
}

size_t SliceNode::GetAxis() const
{
    return _axis;
}
