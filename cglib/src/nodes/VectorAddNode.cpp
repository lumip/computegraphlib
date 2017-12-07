#include "nodes/VectorAddNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

VectorAddNode::VectorAddNode(NodePtr a, NodePtr b)
    : Node(true, false)
    , _summandA(a)
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
        // allow one one input to be a multiple of the other in exactly one dimension (broadcast operation)
        if (!((dimA.xDim == dimB.xDim && (dimA.yDim % dimB.yDim == 0 || dimB.yDim % dimA.yDim == 0)) ||
              (dimA.yDim == dimA.yDim && (dimA.xDim % dimB.xDim == 0 || dimB.xDim % dimA.xDim == 0))))
        {
            throw std::runtime_error("Inputs to VectorAddNode have unmatching dimensions.");
        }
    }
    MemoryDimensions dim { std::max(dimA.yDim, dimB.yDim), std::max(dimA.xDim, dimB.xDim) };
    memoryMap.RegisterNodeMemory(this, dim);
}
