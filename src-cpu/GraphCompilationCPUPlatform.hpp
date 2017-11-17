#ifndef _GRAPH_COMPILATION_CPU_STRATEGY_HPP_
#define _GRAPH_COMPILATION_CPU_STRATEGY_HPP_

#include "GraphCompilationPlatform.hpp"

class CompilationMemoryMap;

class GraphCompilationCPUPlatform : public GraphCompilationPlatform
{
private:
    typedef float* MemoryHandle;
private:
    std::vector<std::unique_ptr<Kernel>> _kernels;
    std::map<ConstNodePtr, std::unique_ptr<float[]>> _bufferMap;
    const CompilationMemoryMap& _dimensionsMap;
public:
    GraphCompilationCPUPlatform(const CompilationMemoryMap& CompilationMemoryMap);
    ~GraphCompilationCPUPlatform();
    void AllocateMemory(const ConstNodePtr node);
    void CopyOutputData(const ConstNodePtr outputNode, DataBuffer& outputBuffer) const;
    void CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer);
    void Evaluate();

    void CompileInputNode(const InputNode* const node);
    void CompileMatrixMultNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const MatrixMultNode* const node);
    void CompileVariableNode(const VariableNode* const node);
    void CompileVectorAddNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const VectorAddNode* const node);
};

#endif
