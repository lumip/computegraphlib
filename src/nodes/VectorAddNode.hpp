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
    Node::ptr _summandA;
    Node::ptr _summandB;
public:
    VectorAddNode(Node::ptr a, Node::ptr b);
    virtual ~VectorAddNode();
    ConstNodeMap GetInputs() const;
    void Compile(void* const context) const;
};

#endif
