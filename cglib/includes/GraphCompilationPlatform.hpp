#ifndef _GRAPH_COMPILATION_PLATFORM_HPP_
#define _GRAPH_COMPILATION_PLATFORM_HPP_

#include "types.hpp"

class ConstMultNode;
class ExpFuncNode;
class InputNode;
class LogFuncNode;
class MatrixMultNode;
class NegateNode;
class ReduceMeanNode;
class ReduceSumNode;
class SliceNode;
class StackNode;
class TransposeNode;
class VariableNode;
class VectorAddNode;
class VectorDivNode;
class VectorMultNode;

class GraphCompilationPlatform
{
public:
    typedef size_t MemoryBufferHandle;
public:
    GraphCompilationPlatform() { }
    virtual ~GraphCompilationPlatform() { }
    virtual MemoryBufferHandle ReserveMemoryBuffer(const ConstNodePtr node) = 0;
    virtual void AssignMemoryBuffer(const ConstNodePtr node, MemoryBufferHandle memory) = 0;
    virtual MemoryBufferHandle GetNodeMemoryBuffer(const ConstNodePtr node) const = 0;
    virtual bool NodeIsAssigned(const ConstNodePtr node) const = 0;
    virtual void CopyOutputData(const ConstNodePtr outputNode, DataBuffer& outputBuffer) const = 0;
    virtual void CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer) = 0;
    virtual void Evaluate() = 0;

    /* todo: remove all graph-specific compilation state from out of this class. make it truly just a "platform" which can be reused */

    virtual void CompileConstMultNode(const ConstMultNode* const node) = 0;
    virtual void CompileExpFuncNode(const ExpFuncNode* const node) = 0;
    virtual void CompileInputNode(const InputNode* const node) = 0;
    virtual void CompileLogFuncNode(const LogFuncNode* const node) = 0;
    virtual void CompileMatrixMultNode(const MatrixMultNode* const node) = 0;
    virtual void CompileNegateNode(const NegateNode* const node) = 0;
    virtual void CompileReduceMeanNode(const ReduceMeanNode* const node) = 0;
    virtual void CompileReduceSumNode(const ReduceSumNode* const node) = 0;
    virtual void CompileSliceNode(const SliceNode* const node) = 0;
    virtual void CompileStackNode(const StackNode* const node) = 0;
    virtual void CompileTransposeNode(const TransposeNode* const node) = 0;
    virtual void CompileVariableNode(const VariableNode* const node) = 0;
    virtual void CompileVectorAddNode(const VectorAddNode* const node) = 0;
    virtual void CompileVectorDivNode(const VectorDivNode* const node) = 0;
    virtual void CompileVectorMultNode(const VectorMultNode* const node) = 0;
};

#endif
