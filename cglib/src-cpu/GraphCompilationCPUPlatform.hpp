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
    struct MemoryBuffer;
private:
    std::vector<std::unique_ptr<Kernel>> _kernels;
    std::vector<std::unique_ptr<float[]>> _memoryBufferLocations;
    std::vector<MemoryBuffer> _memoryBuffers;
    std::map<ConstNodePtr, MemoryBufferHandle> _nodeMemoryAssignments;
    const CompilationMemoryMap& _dimensionsMap;
private:
    MemoryHandle GetMemoryLocation(const ConstNodePtr node) const;
public:
    GraphCompilationCPUPlatform(const CompilationMemoryMap& CompilationMemoryMap);
    ~GraphCompilationCPUPlatform();
    MemoryBufferHandle ReserveMemoryBuffer(const ConstNodePtr node);
    void AssignMemoryBuffer(const ConstNodePtr node, MemoryBufferHandle memory);
    MemoryBufferHandle GetNodeMemoryBuffer(const ConstNodePtr node) const;
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
