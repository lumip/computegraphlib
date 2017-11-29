#include "nodes/StackNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

StackNode::StackNode(const std::vector<NodePtr>& components, size_t axis)
    : _components(std::begin(components), std::end(components))
    , _axis(axis)
{
    for (NodePtr input : components)
    {
        this->SubscribeTo(input);
    }
}

StackNode::~StackNode() { }

Node::ConstNodeList StackNode::GetInputs() const
{
    return _components;
}

std::string StackNode::ToString() const
{
    std::stringstream ss;
    ss << "<StackNode " << (this) << ">";
    return ss.str();
}

bool StackNode::IsInitialized() const
{
    return false;
}

void StackNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileStackNode(this);
}

void StackNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    MemoryDimensions dim = memoryMap.GetNodeMemoryDimensions(_components[0]);
    dim.dims[_axis] *= _components.size();
    memoryMap.RegisterNodeMemory(this, MemoryDimensions(dim));
}