#include "nodes/VectorDivNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

VectorDivNode::VectorDivNode(NodePtr a, NodePtr b)
    : Node(true, false)
    , _inputA(a)
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
        if (!(((dimA.yDim == dimB.yDim) && (dimA.xDim % dimB.xDim == 0)) ||
              ((dimA.xDim == dimB.xDim) && (dimA.yDim % dimB.yDim == 0))))
        {
            throw std::runtime_error("Inputs to VectorDivNode have unmatching dimensions.");
        }
    }
    memoryMap.RegisterNodeMemory(this, dimA);
}
