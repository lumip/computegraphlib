#include "nodes/LogFuncNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

LogFuncNode::LogFuncNode(NodePtr input)
    : _input(input)
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

bool LogFuncNode::IsInitialized() const
{
    return false;
}

void LogFuncNode::Compile(GraphCompilationPlatform& platform) const
{
    throw new std::logic_error("not yet implemented.");
}

void LogFuncNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    const MemoryDimensions dim = memoryMap.GetNodeMemoryDimensions(_input);
    memoryMap.RegisterNodeMemory(this, MemoryDimensions(dim));
}
