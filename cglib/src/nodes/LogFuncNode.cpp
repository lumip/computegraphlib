#include "nodes/LogFuncNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

LogFuncNode::LogFuncNode(NodePtr input)
    : Node(true, false)
    , _input(input)
{
    this->SubscribeTo(input);
}

LogFuncNode::~LogFuncNode() { }

Node::ConstNodeList LogFuncNode::GetInputs() const
{
    return ConstNodeList({ _input });
}

std::string LogFuncNode::ToString() const
{
    std::stringstream ss;
    ss << "<LogFuncNode " << (this) << ">";
    return ss.str();
}

void LogFuncNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileLogFuncNode(this);
}

void LogFuncNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    const MemoryDimensions dim = memoryMap.GetNodeMemoryDimensions(_input);
    memoryMap.RegisterNodeMemory(this, MemoryDimensions(dim));
}
