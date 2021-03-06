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
    std::vector<std::unique_ptr<float[]>> _memoryBufferLocations;
private:
    MemoryHandle GetMemoryLocation(const ConstNodePtr node) const;
public:
    GraphCompilationCPUPlatform(const CompilationMemoryMap& compilationMemoryMap);
    ~GraphCompilationCPUPlatform();
    void AllocateAllMemory();
    void CopyOutputData(const ConstNodePtr outputNode, float* outputBuffer) const;
    void CopyInputData(const ConstNodePtr inputNode, float const* inputBuffer);
    void Evaluate();
    bool IsEvaluating() const;
    void WaitUntilEvaluationFinished() const;
    void WaitUntilDataTransferFinished() const;
    float* GetMappedInputBuffer(std::string const& inputName);

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
