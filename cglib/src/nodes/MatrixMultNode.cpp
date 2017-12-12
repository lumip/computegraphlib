#include "nodes/MatrixMultNode.hpp"

#include <sstream>

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

MatrixMultNode::MatrixMultNode(NodePtr a, NodePtr b)
    : Node(false, false)
    , _a(a)
    , _b(b)
{
    this->SubscribeTo(a);
    this->SubscribeTo(b);
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

void MatrixMultNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileMatrixMultNode(this);
}

void MatrixMultNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
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
