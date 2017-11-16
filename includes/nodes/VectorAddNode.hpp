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
    ConstNodeList GetInputs() const;
    void Compile(MemoryCompilationMap& context, NodeCompiler& nodeCompiler) const;
    void GetMemoryDimensions(MemoryCompilationMap& memoryMap) const;
    std::string ToString() const;
    bool IsInitialized() const;
};

#endif
