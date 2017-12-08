#include "GraphCompilationPlatform.hpp"

GraphCompilationPlatform::GraphCompilationPlatform(const CompilationMemoryMap& compilationMemoryMap)
    : _memoryBuffers()
    , _nodeMemoryAssignments()
    , _dimensionsMap(compilationMemoryMap)
{ }

GraphCompilationPlatform::~GraphCompilationPlatform() { }

GraphCompilationPlatform::MemoryBufferHandle GraphCompilationPlatform::ReserveMemoryBuffer(const ConstNodePtr node)
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

void GraphCompilationPlatform::AssignMemoryBuffer(const ConstNodePtr node, MemoryBufferHandle memory)
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

GraphCompilationPlatform::MemoryBufferHandle GraphCompilationPlatform::GetNodeMemoryBuffer(const ConstNodePtr node) const
{
    return _nodeMemoryAssignments.at(node);
}

bool GraphCompilationPlatform::NodeIsAssigned(const ConstNodePtr node) const
{
    return _nodeMemoryAssignments.find(node) != _nodeMemoryAssignments.end();
}

/*void GraphCompilationPlatform::AllocateAllMemory()
{
    for (size_t i = 0; i < _memoryBuffers.size(); ++i)
    {
        const MemoryBuffer& buffer = _memoryBuffers[i];
        if (buffer.Subscribers.size() > 0)
        {
            AllocateMemory(i, buffer.Dimensions.size());
        }
    }
}*/
