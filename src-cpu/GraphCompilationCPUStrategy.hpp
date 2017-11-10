#ifdef CPU

#ifndef _GRAPH_COMPILATION_CPU_STRATEGY_HPP_
#define _GRAPH_COMPILATION_CPU_STRATEGY_HPP_

#include "GraphCompilationContext.hpp"

class GraphCompilationCPUStrategy : public GraphCompilationTargetStrategy
{
private:
    std::vector<std::unique_ptr<const Kernel>> _kernels;
public:
    GraphCompilationCPUStrategy();
    ~GraphCompilationCPUStrategy();
    NodeMemoryHandle AllocateMemory(size_t size);
    void DeallocateMemory(const NodeMemoryHandle mem);
    void EnqueueKernel(std::unique_ptr<const Kernel>&& kernel);
    void CopyOutputData(const NodeMemoryHandle outputNodeMemory, DataBuffer& outputBuffer, size_t size) const;
    void CopyInputData(const NodeMemoryHandle inputNodeMemory, InputDataBuffer& inputBuffer, size_t size) const;
    void Evaluate() const;
};


#endif

#endif

