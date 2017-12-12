#include "nodes/ExpFuncNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

ExpFuncNode::ExpFuncNode(NodePtr input)
    : Node(true, false)
    , _input(input)
{
    this->SubscribeTo(input);
}

ExpFuncNode::~ExpFuncNode() { }

Node::ConstNodeList ExpFuncNode::GetInputs() const
{
    return ConstNodeList({ _input });
}

std::string ExpFuncNode::ToString() const
{
    std::stringstream ss;
    ss << "<ExpFuncNode " << (this) << ">";
    return ss.str();
}

void ExpFuncNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileExpFuncNode(this);
}

void ExpFuncNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    const MemoryDimensions dim = memoryMap.GetNodeMemoryDimensions(_input);
    memoryMap.RegisterNodeMemory(this, MemoryDimensions(dim));
}
