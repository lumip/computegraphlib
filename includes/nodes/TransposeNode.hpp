#ifndef _TRANSPOSE_NODE_HPP_
#define _TRANSPOSE_NODE_HPP_

#include "nodes/Node.hpp"

class TransposeNode : public Node
{
private:
    const ConstNodePtr _input;
public:
    TransposeNode(NodePtr input);
    virtual ~TransposeNode();
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;
    bool IsInitialized() const;
};

#endif
