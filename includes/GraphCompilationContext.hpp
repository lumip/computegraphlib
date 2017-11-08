#ifndef _GRAPH_COMPILATION_CONTEXT_HPP_
#define _GRAPH_COMPILATION_CONTEXT_HPP_

#include <memory>

#include "types.hpp"
#include "Kernel.hpp"

class GraphCompilationContext
{
public:
    typedef float* NodeMemoryHandle;
    struct NodeMemoryDescriptor
    {
        const GraphCompilationContext::NodeMemoryHandle handle;
        size_t yDim; // todo: decouple compilation from input data
        size_t xDim;
        size_t size;
    };
private:
    void* const _context;
    std::map<ConstNodePtr, NodeMemoryHandle> _memoryMap;
    std::map<NodeMemoryHandle, NodeMemoryDescriptor> _memoryDescriptors;
    const InputDataMap& _inputData;
    std::map<std::string, const NodeMemoryHandle> _inputMemoryMap;
    std::map<std::string, const NodeMemoryHandle> _outputMemoryMap;
public:
    GraphCompilationContext(const InputDataMap& inputData); // todo: decouple compilation from input data
    virtual ~GraphCompilationContext();
    NodeMemoryHandle RegisterMemory(size_t yDim, size_t xDim);
    void AssignNodeMemory(const ConstNodePtr node, const NodeMemoryHandle memoryHandle);
    NodeMemoryDescriptor GetNodeMemoryDescriptor(const ConstNodePtr node) const;
    InputDataBuffer& GetInputDataBuffer(std::string inputName) const;
    void RegisterInputMemory(const std::string inputName, const NodeMemoryHandle memoryHandle);
    void RegisterOutputMemory(const std::string outputName, const NodeMemoryHandle memoryHandle);
    void EnqueueKernel(std::unique_ptr<const Kernel>& kernel);
private:
    NodeMemoryHandle AllocateMemory(size_t size);
    void DeallocateAllMemory();
    void* InitContext(); // todo: this must not survive. hack to have some arbitrary implementation dependent context until proper solution. either subclassing or strategy pattern.
};

#endif
