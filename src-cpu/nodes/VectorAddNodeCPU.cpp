#ifdef CPU
#include "nodes/VectorAddNode.hpp"
#include "GraphCompilationContext.hpp"
#include "Kernel.hpp"

class VectorAddNodeCPUKernel : public Kernel
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
    void Run() const
    {
        for (size_t i = 0; i < _size; ++i)
        {
            _memoryResult[i] = _memoryA[i] + _memoryB[i];
        }
    }
};

void VectorAddNode::Compile(GraphCompilationContext& context) const
{
    NodeMemoryDescriptor memDescA = context.GetNodeMemoryDescriptor(this->_summandA);
    NodeMemoryDescriptor memDescB = context.GetNodeMemoryDescriptor(this->_summandB);
    if (memDescA.size() != memDescB.size())
    {
        throw std::runtime_error("Inputs to VectorAddNode have different size.");
    }
    NodeMemoryHandle memHandle = context.RegisterMemory(memDescA.yDim, memDescA.xDim);
    context.AssignNodeMemory(this, memHandle);
    float* const inputAMemBuffer = reinterpret_cast<float* const>(memDescA.handle);
    float* const inputBMemBuffer = reinterpret_cast<float* const>(memDescB.handle);
    float* const resultMemBuffer = reinterpret_cast<float* const>(memHandle);
    context.EnqueueKernel(std::unique_ptr<const Kernel>(new VectorAddNodeCPUKernel(inputAMemBuffer, inputBMemBuffer, resultMemBuffer, memDescA.size()))); // std::make_unique only since c++14
}

#endif
