#include <CL/cl.h>

#include "nodes/VectorAddNode.hpp"
#include "GraphCompilationContext.hpp"
#include "../GraphCompilationGPUStrategy.hpp"
#include "Kernel.hpp"

/*class VectorAddNodeGPUKernel : public Kernel
{
private:
    const cl_mem _memoryA;
    const cl_mem _memoryB;
    const cl_mem _memoryResult;
    const size_t _size;
public:
    VectorAddNodeGPUKernel(const cl_mem memoryA, const cl_mem memoryB, const cl_mem memoryResult, size_t size)
        : _memoryA(memoryA)
        , _memoryB(memoryB)
        , _memoryResult(memoryResult)
        , _size(size)
    {
    }
    virtual ~VectorAddNodeGPUKernel() { }
    void Run()
    {
        throw std::logic_error("not implemented yet.");
    }
};*/

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
    /*const cl_mem inputAMemBuffer = reinterpret_cast<const cl_mem>(memDescA.handle);
    const cl_mem inputBMemBuffer = reinterpret_cast<const cl_mem>(memDescB.handle);
    const cl_mem resultMemBuffer = reinterpret_cast<const cl_mem>(memDesc.handle);
    context.EnqueueKernel(std::unique_ptr<Kernel>(new VectorAddNodeGPUKernel(inputAMemBuffer, inputBMemBuffer, resultMemBuffer, memDescA.dimensions.size()))); // std::make_unique only since c++14*/
}
