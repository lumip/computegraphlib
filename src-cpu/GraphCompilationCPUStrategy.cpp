#include "GraphCompilationCPUStrategy.hpp"
#include "GraphCompilationContext.hpp"

GraphCompilationCPUStrategy::GraphCompilationCPUStrategy(const MemoryCompilationMap& memoryCompilationMap)
    : _kernels()
    , _bufferMap()
    , _dimensionsMap(memoryCompilationMap)
{
}

GraphCompilationCPUStrategy::~GraphCompilationCPUStrategy() { }

void GraphCompilationCPUStrategy::AllocateMemory(const ConstNodePtr node)
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    std::unique_ptr<float[]> mem(new float[dims.size()]); // consider using std::valarray instead of raw float arrays?
    _bufferMap.emplace(node, std::move(mem));
}

/*void GraphCompilationCPUStrategy::DeallocateMemory(const NodeMemoryHandle mem)
{
    delete[] reinterpret_cast<float* const>(mem);
}*/

/*void GraphCompilationCPUStrategy::EnqueueKernel(std::unique_ptr<Kernel>&& kernel)
{
    _kernels.emplace_back(std::move(kernel));
}*/

void GraphCompilationCPUStrategy::CopyOutputData(const ConstNodePtr outputNode, DataBuffer& outputBuffer) const
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(outputNode);
    size_t size = dims.size();
    outputBuffer.resize(size);
    MemoryHandle nodeMemBuffer = _bufferMap.at(outputNode).get();
    std::copy(nodeMemBuffer, (nodeMemBuffer + size), std::begin(outputBuffer));
}

void GraphCompilationCPUStrategy::CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer)
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    MemoryHandle nodeMemBuffer = _bufferMap.at(inputNode).get();
    std::copy(std::begin(inputBuffer), std::begin(inputBuffer) + dims.size(), nodeMemBuffer);
}

void GraphCompilationCPUStrategy::Evaluate()
{
    for (const std::unique_ptr<Kernel>& kernel : _kernels)
    {
        kernel->Run();
    }
}
