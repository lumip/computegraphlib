#ifndef _VECTOR_ADD_NODE_HPP_
#define _VECTOR_ADD_NODE_HPP_

#include "Node.hpp"

/**
 * Compute sum of two vectors
 */
class VectorAddNode : public Node
{
public:
    typedef std::weak_ptr<VectorAddNode> weak_ptr;
    typedef std::shared_ptr<VectorAddNode> shared_ptr;
    typedef std::unique_ptr<VectorAddNode> unique_ptr;
private:
    NodePtr _summandA;
    NodePtr _summandB;
public:
    VectorAddNode(NodePtr a, NodePtr b);
    virtual ~VectorAddNode();
    ConstNodeMap GetInputs() const;
    std::unique_ptr<const Kernel> Compile(GraphCompilationContext* const context) const;
    std::string ToString() const;
};

#endif
