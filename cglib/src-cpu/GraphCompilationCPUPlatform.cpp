#include "GraphCompilationCPUPlatform.hpp"

#include "CompilationMemoryMap.hpp"
#include "Kernel.hpp"

GraphCompilationCPUPlatform::GraphCompilationCPUPlatform(const CompilationMemoryMap& CompilationMemoryMap)
    : _kernels()
    , _bufferMap()
    , _nodeAssignments()
    , _dimensionsMap(CompilationMemoryMap)
{
}

GraphCompilationCPUPlatform::~GraphCompilationCPUPlatform() { }

GraphCompilationCPUPlatform::MemoryHandle GraphCompilationCPUPlatform::GetMemory(const ConstNodePtr node) const
{
    return _bufferMap[_nodeAssignments.at(node)].get();
}

GraphCompilationPlatform::AbstractMemoryHandle GraphCompilationCPUPlatform::AllocateMemory(const ConstNodePtr node)
{
    AbstractMemoryHandle handle = static_cast<AbstractMemoryHandle>(_bufferMap.size());
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    _bufferMap.emplace_back(new float[dims.size()]); // consider using std::valarray instead of raw float arrays?
    AssignNodeMemory(node, handle);
    return handle;
}

void GraphCompilationCPUPlatform::AssignNodeMemory(const ConstNodePtr node, AbstractMemoryHandle memory)
{
    _nodeAssignments.emplace(node, memory);
}

GraphCompilationPlatform::AbstractMemoryHandle GraphCompilationCPUPlatform::GetNodeMemoryHandle(const ConstNodePtr node) const
{
    return _nodeAssignments.at(node);
}

bool GraphCompilationCPUPlatform::NodeIsAssigned(const ConstNodePtr node) const
{
    return _nodeAssignments.find(node) != _nodeAssignments.end();
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
    MemoryHandle nodeMemBuffer = _bufferMap[_nodeAssignments.at(outputNode)].get();
    std::copy(nodeMemBuffer, (nodeMemBuffer + size), std::begin(outputBuffer));
}

void GraphCompilationCPUPlatform::CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer)
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    MemoryHandle nodeMemBuffer = _bufferMap[_nodeAssignments.at(inputNode)].get();
    std::copy(std::begin(inputBuffer), std::begin(inputBuffer) + dims.size(), nodeMemBuffer);
}

void GraphCompilationCPUPlatform::Evaluate()
{
    for (const std::unique_ptr<Kernel>& kernel : _kernels)
    {
        kernel->Run();
    }
}
