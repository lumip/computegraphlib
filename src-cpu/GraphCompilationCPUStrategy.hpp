#ifndef _GRAPH_COMPILATION_CPU_STRATEGY_HPP_
#define _GRAPH_COMPILATION_CPU_STRATEGY_HPP_

#include "GraphCompilationPlatform.hpp"
#include "Kernel.hpp"

class MemoryCompilationMap;

class GraphCompilationCPUStrategy : public GraphCompilationPlatform
{
private:
    typedef float* MemoryHandle;
private:
    std::vector<std::unique_ptr<Kernel>> _kernels;
    std::map<ConstNodePtr, std::unique_ptr<float[]>> _bufferMap;
    const MemoryCompilationMap& _dimensionsMap;
public:
    GraphCompilationCPUStrategy(const MemoryCompilationMap& memoryCompilationMap);
    ~GraphCompilationCPUStrategy();
    void AllocateMemory(const ConstNodePtr node);
    //void DeallocateMemory(const NodeMemoryHandle mem);
    //void EnqueueKernel(std::unique_ptr<Kernel>&& kernel);
    void CopyOutputData(const ConstNodePtr outputNode, DataBuffer& outputBuffer) const;
    void CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer);
    void Evaluate();

    void CompileInputNode(const InputNode* const node);
    void CompileMatrixMultNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const MatrixMultNode* const node);
    void CompileVariableNode(const VariableNode* const node);
    void CompileVectorAddNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const VectorAddNode* const node);
};

#endif
