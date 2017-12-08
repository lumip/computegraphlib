#ifndef _GRAPH_COMPILATION_CPU_STRATEGY_HPP_
#define _GRAPH_COMPILATION_CPU_STRATEGY_HPP_

#include <memory>

#include "GraphCompilationPlatform.hpp"

class CompilationMemoryMap;
class Kernel;

class GraphCompilationCPUPlatform : public GraphCompilationPlatform
{
private:
    typedef float* MemoryHandle;
private:
    std::vector<std::unique_ptr<Kernel>> _kernels;
    std::vector<std::unique_ptr<float[]>> _bufferMap;
    std::map<ConstNodePtr, AbstractMemoryHandle> _nodeAssignments;
    const CompilationMemoryMap& _dimensionsMap;
private:
    MemoryHandle GetMemory(const ConstNodePtr node) const;
public:
    GraphCompilationCPUPlatform(const CompilationMemoryMap& CompilationMemoryMap);
    ~GraphCompilationCPUPlatform();
    AbstractMemoryHandle AllocateMemory(const ConstNodePtr node);
    void AssignNodeMemory(const ConstNodePtr node, AbstractMemoryHandle memory);
    AbstractMemoryHandle GetNodeMemoryHandle(const ConstNodePtr node) const;
    bool NodeIsAssigned(const ConstNodePtr node) const;
    void CopyOutputData(const ConstNodePtr outputNode, DataBuffer& outputBuffer) const;
    void CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer);
    void Evaluate();

    void CompileConstMultNode(const ConstMultNode* const node);
    void CompileExpFuncNode(const ExpFuncNode* const node);
    void CompileInputNode(const InputNode* const node);
    void CompileLogFuncNode(const LogFuncNode* const node);
    void CompileMatrixMultNode(const MatrixMultNode* const node);
    void CompileNegateNode(const NegateNode* const node);
    void CompileReduceMeanNode(const ReduceMeanNode* const node);
    void CompileReduceSumNode(const ReduceSumNode* const node);
    void CompileSliceNode(const SliceNode* const node);
    void CompileStackNode(const StackNode* const node);
    void CompileTransposeNode(const TransposeNode* const node);
    void CompileVariableNode(const VariableNode* const node);
    void CompileVectorAddNode(const VectorAddNode* const node);
    void CompileVectorDivNode(const VectorDivNode* const node);
    void CompileVectorMultNode(const VectorMultNode* const node);
};

#endif
