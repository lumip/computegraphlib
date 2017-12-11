#include "GraphCompilationCPUPlatform.hpp"

#include <set>

#include "CompilationMemoryMap.hpp"
#include "Kernel.hpp"

GraphCompilationCPUPlatform::GraphCompilationCPUPlatform(const CompilationMemoryMap& compilationMemoryMap)
    :  GraphCompilationPlatform(compilationMemoryMap)
    , _kernels()
    , _memoryBufferLocations()
{ }

GraphCompilationCPUPlatform::~GraphCompilationCPUPlatform() { }

GraphCompilationCPUPlatform::MemoryHandle GraphCompilationCPUPlatform::GetMemoryLocation(const ConstNodePtr node) const
{
    return _memoryBufferLocations[GetNodeMemoryBuffer(node)].get();
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

void GraphCompilationCPUPlatform::CopyOutputData(const ConstNodePtr outputNode, float* outputBuffer) const
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(outputNode);
    size_t size = dims.size();
    MemoryHandle nodeMemBuffer = GetMemoryLocation(outputNode);
    std::copy(nodeMemBuffer, (nodeMemBuffer + size), outputBuffer);
}

void GraphCompilationCPUPlatform::CopyInputData(const ConstNodePtr inputNode, float const* inputBuffer)
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    MemoryHandle nodeMemBuffer = GetMemoryLocation(inputNode);
    if (inputBuffer != nodeMemBuffer)
    {
        std::copy(inputBuffer, inputBuffer + dims.size(), nodeMemBuffer);
    }
}

void GraphCompilationCPUPlatform::Evaluate()
{
    for (const std::unique_ptr<Kernel>& kernel : _kernels)
    {
        kernel->Run();
    }
}

bool GraphCompilationCPUPlatform::IsEvaluating() const
{
    return false;
}

void GraphCompilationCPUPlatform::WaitUntilEvaluationFinished() const { }

void GraphCompilationCPUPlatform::WaitUntilDataTransferFinished() const { }

float* GraphCompilationCPUPlatform::GetMappedInputBuffer(std::string const& inputName)
{
    ConstNodePtr node = nullptr;
    try
    {
        node = _dimensionsMap.GetInputNode(inputName);
    }
    catch (std::out_of_range&)
    {
        node = _dimensionsMap.GetVariableNode(inputName);
    }
    MemoryHandle nodeMemBuffer = GetMemoryLocation(node);
    return nodeMemBuffer;
}
