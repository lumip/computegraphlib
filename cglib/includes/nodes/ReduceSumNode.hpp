#ifndef _REDUCE_SUM_NODE_HPP_
#define _REDUCE_SUM_NODE_HPP_

#include "nodes/Node.hpp"

class ReduceSumNode : public Node
{
private:
    const ConstNodePtr _input;
    const size_t _axis;
public:
    ReduceSumNode(NodePtr input, size_t axis);
    virtual ~ReduceSumNode();
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;
    bool IsInitialized() const;

    size_t GetAxis() const;
};

#endif
