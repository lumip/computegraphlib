#include <sstream>

#include "nodes/MatrixMultNode.hpp"

MatrixMultNode::MatrixMultNode(NodePtr a, NodePtr b)
    : _a(a)
    , _b(b)
{

}

MatrixMultNode::~MatrixMultNode() { }

Node::ConstNodeList MatrixMultNode::GetInputs() const
{
    return ConstNodeList({ _a, _b });
}

std::string MatrixMultNode::ToString() const
{
    std::stringstream ss;
    ss << "<MatrixMultNode " << (this) << ">";
    return ss.str();
}

bool MatrixMultNode::IsInitialized() const
{
    return false;
}

void MatrixMultNode::Compile(GraphCompilationContext& context, NodeCompiler& nodeCompiler) const
{
    NodeMemoryDescriptor memDescA = context.GetNodeMemoryDescriptor(this->_a);
    NodeMemoryDescriptor memDescB = context.GetNodeMemoryDescriptor(this->_b);
    if (memDescA.dimensions.xDim != memDescB.dimensions.yDim)
    {
        throw new std::invalid_argument("Matrix dimensions do not agree.");
    }
    auto m = memDescA.dimensions.yDim;
    auto d = memDescA.dimensions.xDim; // == memDescB.yDim;
    auto n = memDescB.dimensions.xDim;
    const NodeMemoryDescriptor memDesc = context.AllocateMemory({m, n});
    context.AssignNodeMemory(this, memDesc.handle);
    context.EnqueueKernel(nodeCompiler.CompileMatrixMultNode(memDescA, memDescB, memDesc));
}

MemoryDimensions MatrixMultNode::GetMemoryDimensions(const InputDimensionsMap& inputDimensions, const std::map<ConstNodePtr, MemoryDimensions>& nodeMemoryDimensions) const
{
    const MemoryDimensions dimA = nodeMemoryDimensions.at(_a);
    const MemoryDimensions dimB = nodeMemoryDimensions.at(_b);
    if (dimA.xDim != dimB.yDim)
    {
        throw new std::invalid_argument("Matrix dimensions do not agree.");
    }
    auto m = dimA.yDim;
    auto n = dimB.xDim;
    return MemoryDimensions({m, n});
}
