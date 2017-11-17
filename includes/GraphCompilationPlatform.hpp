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
    virtual void CopyOutputData(const ConstNodePtr outputNode, DataBuffer& outputBuffer) const = 0;
    virtual void CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer) = 0;
    virtual void Evaluate() = 0;
    /* todo: think about identifying memory by some handle instead of by node. this would allow
     * for abstracting lookup of memory dimensions (which is currently duplicated in the subclasses).
     * platform specific implementations could then really focus on just providing the platform
     * dependent code without duplicating the code looking up memory sizes etc
     */
};

#endif
