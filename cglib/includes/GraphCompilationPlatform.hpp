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
    typedef size_t AbstractMemoryHandle;
public:
    GraphCompilationPlatform() { }
    virtual ~GraphCompilationPlatform() { }
    virtual AbstractMemoryHandle AllocateMemory(const ConstNodePtr node) = 0;
    virtual void AssignNodeMemory(const ConstNodePtr node, AbstractMemoryHandle memory) = 0;
    virtual void CopyOutputData(const ConstNodePtr outputNode, DataBuffer& outputBuffer) const = 0;
    virtual void CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer) = 0;
    virtual void Evaluate() = 0;
    /* todo: think about identifying memory by some handle instead of by node. this would allow
     * for abstracting lookup of memory dimensions (which is currently duplicated in the subclasses).
     * platform specific implementations could then really focus on just providing the platform
     * dependent code without duplicating the code looking up memory sizes etc
     */

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
