#include <sstream>

#include "nodes/VectorAddNode.hpp"
#include "GraphCompilationContext.hpp"

VectorAddNode::VectorAddNode(NodePtr a, NodePtr b)
    : Node(), 
    _summandA(a), 
    _summandB(b)
{
    this->SubscribeTo(_summandA);
    this->SubscribeTo(_summandB);
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

void VectorAddNode::Compile(MemoryCompilationMap& context, NodeCompiler& nodeCompiler) const
{
    nodeCompiler.CompileVectorAddNode(_summandA, _summandB, this);
    //context.EnqueueKernel(nodeCompiler.CompileVectorAddNode(memDescA, memDescB, memDesc));
}

void VectorAddNode::GetMemoryDimensions(MemoryCompilationMap& memoryMap) const
{
    const MemoryDimensions dimA = memoryMap.GetNodeMemoryDimensions(_summandA);
    const MemoryDimensions dimB = memoryMap.GetNodeMemoryDimensions(_summandB);
    if (dimA != dimB)
    {
        throw std::runtime_error("Inputs to VectorAddNode have different dimensions.");
    }
    memoryMap.RegisterNodeMemory(this, dimA);
}
