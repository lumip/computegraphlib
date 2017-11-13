#ifdef GPU

#ifndef _GRAPH_COMPILATION_GPU_STRATEGY_HPP_
#define _GRAPH_COMPILATION_GPU_STRATEGY_HPP_

#include "GraphCompilationContext.hpp"

class GraphCompilationGPUStrategy : public GraphCompilationTargetStrategy
{
public:
    GraphCompilationGPUStrategy();
    ~GraphCompilationGPUStrategy();
    NodeMemoryHandle AllocateMemory(size_t size);
    void DeallocateMemory(const NodeMemoryHandle mem);
    void EnqueueKernel(std::unique_ptr<Kernel>&& kernel);
    void CopyOutputData(const NodeMemoryHandle outputNodeMemory, DataBuffer& outputBuffer, size_t size) const;
    void CopyInputData(const NodeMemoryHandle inputNodeMemory, InputDataBuffer& inputBuffer, size_t size);
    void Evaluate(const InputDataMap& inputData);
};


#endif

#endif

