#include "nodes/VectorMultNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

VectorMultNode::VectorMultNode(NodePtr a, NodePtr b)
    : _inputA(a)
    , _inputB(b)
{
    this->SubscribeTo(a);
    this->SubscribeTo(b);
}

VectorMultNode::~VectorMultNode() { }

Node::ConstNodeList VectorMultNode::GetInputs() const
{
    return ConstNodeList({ _inputA, _inputB });
}

std::string VectorMultNode::ToString() const
{
    std::stringstream ss;
    ss << "<VectorMultNode " << (this) << ">";
    return ss.str();
}

bool VectorMultNode::IsInitialized() const
{
    return false;
}

void VectorMultNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileVectorMultNode(this);
}

void VectorMultNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    const MemoryDimensions dimA = memoryMap.GetNodeMemoryDimensions(_inputA);
    const MemoryDimensions dimB = memoryMap.GetNodeMemoryDimensions(_inputB);
    if (dimA != dimB)
    {
        throw std::runtime_error("Inputs to VectorDivNode have different dimensions.");
    }
    memoryMap.RegisterNodeMemory(this, dimA);
}
