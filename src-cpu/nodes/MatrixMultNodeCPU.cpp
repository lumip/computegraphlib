#include "nodes/MatrixMultNode.hpp"
#include "GraphCompilationContext.hpp"
#include "Kernel.hpp"

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
    const NodeMemoryDescriptor memDesc = context.RegisterMemory({m, n});
    context.AssignNodeMemory(this, memDesc.handle);
    context.EnqueueKernel(nodeCompiler.CompileMatrixMultNode(memDescA, memDescB, memDesc));
    /*const float* const inputAMemBuffer = reinterpret_cast<const float* const>(memDescA.handle);
    const float* const inputBMemBuffer = reinterpret_cast<const float* const>(memDescB.handle);
    context.EnqueueKernel(std::unique_ptr<Kernel>(new MatrixMultCPUKernel(inputAMemBuffer, inputBMemBuffer, memDesc, d))); // std::make_unique only since c++14*/
}
