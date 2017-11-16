#ifndef _GRAPH_COMPILATION_PLATFORM_HPP_
#define _GRAPH_COMPILATION_PLATFORM_HPP_

#include "types.hpp"
#include "NodeCompiler.hpp"

class GraphCompilationPlatform : public NodeCompiler
{
public:
    GraphCompilationPlatform() { }
    virtual ~GraphCompilationPlatform() { }
    virtual void AllocateMemory(const ConstNodePtr node) = 0;
    //virtual void DeallocateMemory(const NodeMemoryHandle mem) = 0;
    //virtual void EnqueueKernel(std::unique_ptr<Kernel>&& kernel) = 0;
    virtual void CopyOutputData(const ConstNodePtr outputNode, DataBuffer& outputBuffer) const = 0;
    virtual void CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer) = 0;
    virtual void Evaluate() = 0;
};

#endif
