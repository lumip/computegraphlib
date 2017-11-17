#ifndef _MATRIX_MULT_NODE_HPP_
#define _MATRIX_MULT_NODE_HPP_

#include "Node.hpp"

class MatrixMultNode : public Node
{
private:
    ConstNodePtr _a;
    ConstNodePtr _b;
public:
    MatrixMultNode(NodePtr a, NodePtr b);
    virtual ~MatrixMultNode();
    ConstNodeList GetInputs() const;
    void Compile(CompilationMemoryMap& context, NodeCompiler& nodeCompiler) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;
    bool IsInitialized() const;
};

#endif
