#ifdef CPU

#include "GraphCompilationCPUStrategy.hpp"

GraphCompilationCPUStrategy::GraphCompilationCPUStrategy()
    : _kernels()
{
}

GraphCompilationCPUStrategy::~GraphCompilationCPUStrategy() { }

NodeMemoryHandle GraphCompilationCPUStrategy::AllocateMemory(size_t size)
{
    return reinterpret_cast<NodeMemoryHandle>(new float[size]); // consider using std::valarray instead of raw float arrays?
}

void GraphCompilationCPUStrategy::DeallocateMemory(const NodeMemoryHandle mem)
{
    delete[] reinterpret_cast<float* const>(mem);
}

void GraphCompilationCPUStrategy::EnqueueKernel(std::unique_ptr<Kernel>&& kernel)
{
    _kernels.emplace_back(std::move(kernel));
}

void GraphCompilationCPUStrategy::CopyOutputData(const NodeMemoryHandle outputNodeMemory, DataBuffer& outputBuffer, size_t size) const
{
    float* const nodeMemBuffer = reinterpret_cast<float* const>(outputNodeMemory);
    std::copy(nodeMemBuffer, (nodeMemBuffer + size), std::begin(outputBuffer));
}

void GraphCompilationCPUStrategy::CopyInputData(const NodeMemoryHandle inputNodeMemory, InputDataBuffer& inputBuffer, size_t size)
{
    float* const nodeMemBuffer = reinterpret_cast<float* const>(inputNodeMemory);
    std::copy(std::begin(inputBuffer), std::begin(inputBuffer) + size, nodeMemBuffer);
}

void GraphCompilationCPUStrategy::Evaluate(const InputDataMap& inputData)
{
    for (const std::unique_ptr<Kernel>& kernel : _kernels)
    {
        kernel->Run();
    }
}

#endif
