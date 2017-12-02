#include "GraphCompilationCPUPlatform.hpp"

#include "CompilationMemoryMap.hpp"
#include "Kernel.hpp"

GraphCompilationCPUPlatform::GraphCompilationCPUPlatform(const CompilationMemoryMap& CompilationMemoryMap)
    : _kernels()
    , _bufferMap()
    , _dimensionsMap(CompilationMemoryMap)
{
}

GraphCompilationCPUPlatform::~GraphCompilationCPUPlatform() { }

void GraphCompilationCPUPlatform::AllocateMemory(const ConstNodePtr node)
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

void GraphCompilationCPUPlatform::CopyOutputData(const ConstNodePtr outputNode, DataBuffer& outputBuffer) const
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(outputNode);
    size_t size = dims.size();
    outputBuffer.resize(size);
    MemoryHandle nodeMemBuffer = _bufferMap.at(outputNode).get();
    std::copy(nodeMemBuffer, (nodeMemBuffer + size), std::begin(outputBuffer));
}

void GraphCompilationCPUPlatform::CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer)
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    MemoryHandle nodeMemBuffer = _bufferMap.at(inputNode).get();
    std::copy(std::begin(inputBuffer), std::begin(inputBuffer) + dims.size(), nodeMemBuffer);
}

void GraphCompilationCPUPlatform::Evaluate()
{
    for (const std::unique_ptr<Kernel>& kernel : _kernels)
    {
        kernel->Run();
    }
}
