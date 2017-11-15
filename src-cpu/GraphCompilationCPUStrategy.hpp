#ifndef _GRAPH_COMPILATION_CPU_STRATEGY_HPP_
#define _GRAPH_COMPILATION_CPU_STRATEGY_HPP_

#include "GraphCompilationContext.hpp"

class GraphCompilationCPUStrategy : public GraphCompilationTargetStrategy
{
private:
    std::vector<std::unique_ptr<Kernel>> _kernels;
public:
    GraphCompilationCPUStrategy();
    ~GraphCompilationCPUStrategy();
    NodeMemoryHandle AllocateMemory(size_t size);
    void DeallocateMemory(const NodeMemoryHandle mem);
    void EnqueueKernel(std::unique_ptr<Kernel>&& kernel);
    void CopyOutputData(const NodeMemoryHandle outputNodeMemory, DataBuffer& outputBuffer, size_t size) const;
    void CopyInputData(const NodeMemoryHandle inputNodeMemory, InputDataBuffer& inputBuffer, size_t size);
    void Evaluate(const std::vector<std::pair<const NodeMemoryDescriptor, InputDataBuffer&>>& inputData);
};

#endif
