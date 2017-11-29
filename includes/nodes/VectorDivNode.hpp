#ifndef _VECTOR_DIV_NODE_HPP_
#define _VECTOR_DIV_NODE_HPP_

#include "Node.hpp"

class VectorDivNode : public Node
{
private:
    const ConstNodePtr _inputA;
    const ConstNodePtr _inputB;
public:
    VectorDivNode(NodePtr a, NodePtr b);
    virtual ~VectorDivNode();
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;
    bool IsInitialized() const;
};

#endif
