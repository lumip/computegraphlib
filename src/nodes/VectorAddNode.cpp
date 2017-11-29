#include "nodes/VectorAddNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

VectorAddNode::VectorAddNode(NodePtr a, NodePtr b)
    : _summandA(a)
    , _summandB(b)
{
    this->SubscribeTo(a);
    this->SubscribeTo(b);
}

VectorAddNode::~VectorAddNode() { }

Node::ConstNodeList VectorAddNode::GetInputs() const
{
    return ConstNodeList({ _summandA, _summandB });
}

std::string VectorAddNode::ToString() const
{
    std::stringstream ss;
    ss << "<VectorAddNode " << (this) << ">";
    return ss.str();
}

bool VectorAddNode::IsInitialized() const
{
    return false;
}

void VectorAddNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileVectorAddNode(this);
}

void VectorAddNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    const MemoryDimensions dimA = memoryMap.GetNodeMemoryDimensions(_summandA);
    const MemoryDimensions dimB = memoryMap.GetNodeMemoryDimensions(_summandB);
    if (dimA != dimB)
    {
        throw std::runtime_error("Inputs to VectorAddNode have different dimensions.");
    }
    memoryMap.RegisterNodeMemory(this, dimA);
}
