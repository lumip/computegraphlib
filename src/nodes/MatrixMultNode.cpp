#include <sstream>

#include "nodes/MatrixMultNode.hpp"
#include "GraphCompilationContext.hpp"

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

void MatrixMultNode::Compile(MemoryCompilationMap& context, NodeCompiler& nodeCompiler) const
{
    nodeCompiler.CompileMatrixMultNode(_a, _b, this);
    //context.EnqueueKernel(nodeCompiler.CompileMatrixMultNode(memDescA, memDescB, memDesc));
}

void MatrixMultNode::GetMemoryDimensions(MemoryCompilationMap& memoryMap) const
{
    const MemoryDimensions dimA = memoryMap.GetNodeMemoryDimensions(_a);
    const MemoryDimensions dimB = memoryMap.GetNodeMemoryDimensions(_b);
    if (dimA.xDim != dimB.yDim)
    {
        throw new std::invalid_argument("Matrix dimensions do not agree.");
    }
    auto m = dimA.yDim;
    auto n = dimB.xDim;
    memoryMap.RegisterNodeMemory(this, MemoryDimensions({m, n}));
}
