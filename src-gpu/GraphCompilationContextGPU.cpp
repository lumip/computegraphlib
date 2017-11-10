#ifdef GPU
#include "GraphCompilationContext.hpp"

class GraphCompilationGPUStrategy : public GraphCompilationTargetStrategy
{
private:
    std::vector<std::unique_ptr<const Kernel>> _kernels;
public:
    GraphCompilationGPUStrategy() { }

    ~GraphCompilationGPUStrategy() { }

    NodeMemoryHandle AllocateMemory(size_t size)
    {
        throw std::logic_error("not implemented yet.");
    }

    void DeallocateMemory(const NodeMemoryHandle mem)
    {
        throw std::logic_error("not implemented yet.");
    }

    void EnqueueKernel(std::unique_ptr<const Kernel>&& kernel)
    {
        throw std::logic_error("not implemented yet.");
    }

    void CopyOutputData(const NodeMemoryHandle outputNodeMemory, DataBuffer& outputBuffer, size_t size) const
    {
        throw std::logic_error("not implemented yet.");
    }

    void CopyInputData(const NodeMemoryHandle inputNodeMemory, InputDataBuffer& inputBuffer, size_t size) const
    {
        throw std::logic_error("not implemented yet.");
    }

    void Evaluate() const
    {
        for (const std::unique_ptr<const Kernel>& kernel : _kernels)
        {
            kernel->Run();
        }
    }
};

std::unique_ptr<GraphCompilationTargetStrategy> GraphCompilationContext::InitStrategy()
{
    return std::unique_ptr<GraphCompilationTargetStrategy>(new GraphCompilationGPUStrategy);
}

#endif
