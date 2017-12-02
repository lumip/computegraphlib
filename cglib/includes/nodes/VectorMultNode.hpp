#ifndef _VECTOR_MULT_NODE_HPP_
#define _VECTOR_MULT_NODE_HPP_

#include "Node.hpp"

class VectorMultNode : public Node
{
private:
    const ConstNodePtr _inputA;
    const ConstNodePtr _inputB;
public:
    VectorMultNode(NodePtr a, NodePtr b);
    virtual ~VectorMultNode();
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;
    bool IsInitialized() const;
};

#endif
