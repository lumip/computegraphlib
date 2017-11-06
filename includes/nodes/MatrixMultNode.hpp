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
    ConstNodeMap GetInputs() const;
    std::unique_ptr<const Kernel> Compile(GraphCompilationContext* const context) const;
    std::string ToString() const;
};

#endif
