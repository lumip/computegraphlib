#ifndef _GRAPH_COMPILATION_PLATFORM_HPP_
#define _GRAPH_COMPILATION_PLATFORM_HPP_

#include "types.hpp"

class InputNode;
class MatrixMultNode;
class VariableNode;
class VectorAddNode;

class GraphCompilationPlatform
{
public:
    GraphCompilationPlatform() { }
    virtual ~GraphCompilationPlatform() { }
    virtual void AllocateMemory(const ConstNodePtr node) = 0;
    virtual void CopyOutputData(const ConstNodePtr outputNode, DataBuffer& outputBuffer) const = 0;
    virtual void CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer) = 0;
    virtual void Evaluate() = 0;
    /* todo: think about identifying memory by some handle instead of by node. this would allow
     * for abstracting lookup of memory dimensions (which is currently duplicated in the subclasses).
     * platform specific implementations could then really focus on just providing the platform
     * dependent code without duplicating the code looking up memory sizes etc
     */

    /* todo: remove all graph-specific compilation state from out of this class. make it truly just a "platform" which can be reused */

    virtual void CompileInputNode(const InputNode* const node) = 0;
    virtual void CompileMatrixMultNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const MatrixMultNode* const node) = 0;
    virtual void CompileVariableNode(const VariableNode* const node) = 0;
    virtual void CompileVectorAddNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const VectorAddNode* const node) = 0;
};

#endif
