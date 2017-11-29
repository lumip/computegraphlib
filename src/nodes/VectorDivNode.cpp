#include "nodes/VectorDivNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

VectorDivNode::VectorDivNode(NodePtr a, NodePtr b)
    : _inputA(a)
    , _inputB(b)
{
    this->SubscribeTo(a);
    this->SubscribeTo(b);
}

VectorDivNode::~VectorDivNode() { }

Node::ConstNodeList VectorDivNode::GetInputs() const
{
    return ConstNodeList({ _inputA, _inputB });
}

std::string VectorDivNode::ToString() const
{
    std::stringstream ss;
    ss << "<VectorDivNode " << (this) << ">";
    return ss.str();
}

bool VectorDivNode::IsInitialized() const
{
    return false;
}

void VectorDivNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileVectorDivNode(this);
}

void VectorDivNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    const MemoryDimensions dimA = memoryMap.GetNodeMemoryDimensions(_inputA);
    const MemoryDimensions dimB = memoryMap.GetNodeMemoryDimensions(_inputB);
    if (dimA != dimB)
    {
        throw std::runtime_error("Inputs to VectorDivNode have different dimensions.");
    }
    memoryMap.RegisterNodeMemory(this, dimA);
}
