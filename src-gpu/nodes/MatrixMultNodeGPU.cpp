#include <CL/cl.h>

#include "nodes/MatrixMultNode.hpp"
#include "GraphCompilationContext.hpp"
#include "Kernel.hpp"

class MatrixMultGPUKernel : public Kernel
{
private:
    const cl_mem _memoryA;
    const cl_mem _memoryB;
    const cl_mem _memoryResult;
    const size_t _m;
    const size_t _n;
    const size_t _d;
public:
    MatrixMultGPUKernel(const cl_mem memoryA, const cl_mem memoryB, const NodeMemoryDescriptor memoryResDesc, const size_t vectorDim)
        : _memoryA(memoryA)
        , _memoryB(memoryB)
        , _memoryResult(reinterpret_cast<cl_mem>(memoryResDesc.handle))
        , _m(memoryResDesc.dimensions.yDim)
        , _n(memoryResDesc.dimensions.xDim)
        , _d(vectorDim)
    { }

    virtual ~MatrixMultGPUKernel() { }

    void Run()
    {
        throw std::logic_error("not implemented yet.");
    }
};

void MatrixMultNode::Compile(GraphCompilationContext& context) const
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
    const cl_mem inputAMemBuffer = reinterpret_cast<const cl_mem>(memDescA.handle);
    const cl_mem inputBMemBuffer = reinterpret_cast<const cl_mem>(memDescB.handle);
    context.EnqueueKernel(std::unique_ptr<Kernel>(new MatrixMultGPUKernel(inputAMemBuffer, inputBMemBuffer, memDesc, d))); // std::make_unique only since c++14
}
