#include "nodes/VectorMultNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

VectorMultNode::VectorMultNode(NodePtr a, NodePtr b)
    : Node(true, false)
    , _inputA(a)
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
        // allow one one input to be a multiple of the other in exactly one dimension (broadcast operation)
        if (!((dimA.xDim == dimB.xDim && (dimA.yDim % dimB.yDim == 0 || dimB.yDim % dimA.yDim == 0)) ||
              (dimA.yDim == dimA.yDim && (dimA.xDim % dimB.xDim == 0 || dimB.xDim % dimA.xDim == 0))))
        {
            throw std::runtime_error("Inputs to VectorDivNode have unmatching dimensions.");
        }
    }
    MemoryDimensions dim { std::max(dimA.yDim, dimB.yDim), std::max(dimA.xDim, dimB.xDim) };
    memoryMap.RegisterNodeMemory(this, dim);
}
