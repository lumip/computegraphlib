#include "GraphCompilationCPUPlatform.hpp"

#include <set>

#include "CompilationMemoryMap.hpp"
#include "Kernel.hpp"

struct GraphCompilationCPUPlatform::MemoryBuffer
{
    MemoryDimensions Dimensions;
    std::set<ConstNodePtr> Subscribers;
};

GraphCompilationCPUPlatform::GraphCompilationCPUPlatform(const CompilationMemoryMap& CompilationMemoryMap)
    : _kernels()
    , _memoryBufferLocations()
    , _memoryBuffers()
    , _nodeMemoryAssignments()
    , _dimensionsMap(CompilationMemoryMap)
{
}

GraphCompilationCPUPlatform::~GraphCompilationCPUPlatform() { }

GraphCompilationCPUPlatform::MemoryHandle GraphCompilationCPUPlatform::GetMemoryLocation(const ConstNodePtr node) const
{
    return _memoryBufferLocations[_nodeMemoryAssignments.at(node)].get();
}

GraphCompilationPlatform::MemoryBufferHandle GraphCompilationCPUPlatform::ReserveMemoryBuffer(const ConstNodePtr node)
{
    MemoryBufferHandle handle = static_cast<MemoryBufferHandle>(_memoryBuffers.size());
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    MemoryBuffer buffer;
    buffer.Dimensions = dims;
    buffer.Subscribers.insert(node);
    _memoryBuffers.push_back(buffer);
    AssignMemoryBuffer(node, handle);
    return handle;
}

void GraphCompilationCPUPlatform::AssignMemoryBuffer(const ConstNodePtr node, MemoryBufferHandle memory)
{
    if (NodeIsAssigned(node))
    {
        // if the node is already assigned to a memory handle, we will have two merge the old and the new buffers:
        // - move all subscribers from the previous to the new buffer and clear the previous buffer (abandon it)
        MemoryBuffer& oldBuffer = _memoryBuffers[GetNodeMemoryBuffer(node)];
        MemoryBuffer& newBuffer = _memoryBuffers[memory];
        newBuffer.Subscribers.insert(oldBuffer.Subscribers.cbegin(), oldBuffer.Subscribers.cend());
        for (ConstNodePtr n : oldBuffer.Subscribers)
        {
            _nodeMemoryAssignments[n] = memory;
        }
        oldBuffer.Subscribers.clear();
        oldBuffer.Dimensions = {0, 0};
    }
    else
    {
        _nodeMemoryAssignments[node] = memory;
        _memoryBuffers[memory].Subscribers.insert(node);
    }
}

GraphCompilationPlatform::MemoryBufferHandle GraphCompilationCPUPlatform::GetNodeMemoryBuffer(const ConstNodePtr node) const
{
    return _nodeMemoryAssignments.at(node);
}

bool GraphCompilationCPUPlatform::NodeIsAssigned(const ConstNodePtr node) const
{
    return _nodeMemoryAssignments.find(node) != _nodeMemoryAssignments.end();
}

void GraphCompilationCPUPlatform::AllocateAllMemory()
{
    _memoryBufferLocations.resize(_memoryBuffers.size());
    for (size_t i = 0; i < _memoryBuffers.size(); ++i)
    {
        const MemoryBuffer& buffer = _memoryBuffers[i];
        if (buffer.Subscribers.size() > 0)
        {
            _memoryBufferLocations[i] = std::unique_ptr<float[]>(new float[buffer.Dimensions.size()]);
        }
    }
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
    MemoryHandle nodeMemBuffer = _memoryBufferLocations[_nodeMemoryAssignments.at(outputNode)].get();
    std::copy(nodeMemBuffer, (nodeMemBuffer + size), std::begin(outputBuffer));
}

void GraphCompilationCPUPlatform::CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer)
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    MemoryHandle nodeMemBuffer = _memoryBufferLocations[_nodeMemoryAssignments.at(inputNode)].get();
    std::copy(std::begin(inputBuffer), std::begin(inputBuffer) + dims.size(), nodeMemBuffer);
}

void GraphCompilationCPUPlatform::Evaluate()
{
    for (const std::unique_ptr<Kernel>& kernel : _kernels)
    {
        kernel->Run();
    }
}
