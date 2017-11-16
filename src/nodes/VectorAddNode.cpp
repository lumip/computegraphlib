#include "nodes/VectorAddNode.hpp"
#include <sstream>

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

void VectorAddNode::Compile(GraphCompilationContext& context, NodeCompiler& nodeCompiler) const
{
    NodeMemoryDescriptor memDescA = context.GetNodeMemoryDescriptor(this->_summandA);
    NodeMemoryDescriptor memDescB = context.GetNodeMemoryDescriptor(this->_summandB);
    if (memDescA.dimensions.size() != memDescB.dimensions.size())
    {
        throw std::runtime_error("Inputs to VectorAddNode have different dimension.");
    }
    const NodeMemoryDescriptor memDesc= context.RegisterMemory({memDescA.dimensions.yDim, memDescA.dimensions.xDim});
    context.AssignNodeMemory(this, memDesc.handle);
    context.EnqueueKernel(nodeCompiler.CompileVectorAddNode(memDescA, memDescB, memDesc));
}
