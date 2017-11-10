#ifdef CPU
#include "GraphCompilationContext.hpp"

class GraphCompilationCPUStrategy : public GraphCompilationTargetStrategy
{
private:
    std::vector<std::unique_ptr<const Kernel>> _kernels;
public:
    GraphCompilationCPUStrategy() { }

    ~GraphCompilationCPUStrategy() { }

    NodeMemoryHandle AllocateMemory(size_t size)
    {
        return reinterpret_cast<NodeMemoryHandle>(new float[size]); // consider using std::valarray instead of raw float arrays?
    }

    void DeallocateMemory(const NodeMemoryHandle mem)
    {
        delete[] reinterpret_cast<float* const>(mem);
    }

    void EnqueueKernel(std::unique_ptr<const Kernel>&& kernel)
    {
        _kernels.emplace_back(std::move(kernel));
    }

    void CopyOutputData(const NodeMemoryHandle outputNodeMemory, DataBuffer& outputBuffer, size_t size) const
    {
        float* const nodeMemBuffer = reinterpret_cast<float* const>(outputNodeMemory);
        std::copy(nodeMemBuffer, (nodeMemBuffer + size), std::begin(outputBuffer));
    }

    void CopyInputData(const NodeMemoryHandle inputNodeMemory, InputDataBuffer& inputBuffer, size_t size) const
    {
        float* const nodeMemBuffer = reinterpret_cast<float* const>(inputNodeMemory);
        std::copy(std::begin(inputBuffer), std::begin(inputBuffer) + size, nodeMemBuffer);
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
    return std::unique_ptr<GraphCompilationTargetStrategy>(new GraphCompilationCPUStrategy);
}

#endif
