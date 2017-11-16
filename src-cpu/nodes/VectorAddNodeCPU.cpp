#include "nodes/VectorAddNode.hpp"
#include "GraphCompilationContext.hpp"
#include "Kernel.hpp"

/*class VectorAddNodeCPUKernel : public Kernel
{
private:
    const float* const _memoryA;
    const float* const _memoryB;
    float* const _memoryResult;
    const size_t _size;
public:
    VectorAddNodeCPUKernel(const float* const memoryA, const float* const memoryB, float* const memoryResult, size_t size)
        : _memoryA(memoryA)
        , _memoryB(memoryB)
        , _memoryResult(memoryResult)
        , _size(size)
    {
    }
    virtual ~VectorAddNodeCPUKernel() { }
    void Run()
    {
        for (size_t i = 0; i < _size; ++i)
        {
            _memoryResult[i] = _memoryA[i] + _memoryB[i];
        }
    }
};*/

/*void VectorAddNode::Compile(GraphCompilationContext& context, NodeCompiler& nodeCompiler) const
{
    NodeMemoryDescriptor memDescA = context.GetNodeMemoryDescriptor(this->_summandA);
    NodeMemoryDescriptor memDescB = context.GetNodeMemoryDescriptor(this->_summandB);
    if (memDescA.dimensions.size() != memDescB.dimensions.size())
    {
        throw std::runtime_error("Inputs to VectorAddNode have different dimension.");
    }
    const NodeMemoryDescriptor memDesc= context.RegisterMemory({memDescA.dimensions.yDim, memDescA.dimensions.xDim});
    context.AssignNodeMemory(this, memDesc.handle);
    const float* const inputAMemBuffer = reinterpret_cast<const float* const>(memDescA.handle);
    const float* const inputBMemBuffer = reinterpret_cast<const float* const>(memDescB.handle);
    float* const resultMemBuffer = reinterpret_cast<float* const>(memDesc.handle);
    context.EnqueueKernel(std::unique_ptr<Kernel>(new VectorAddNodeCPUKernel(inputAMemBuffer, inputBMemBuffer, resultMemBuffer, memDescA.dimensions.size()))); // std::make_unique only since c++14
}*/

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
