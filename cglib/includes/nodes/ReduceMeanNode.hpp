#ifndef _REDUCE_MEAN_NODE_HPP_
#define _REDUCE_MEAN_NODE_HPP_

#include "nodes/Node.hpp"

class ReduceMeanNode : public Node
{
private:
    const ConstNodePtr _input;
    const size_t _axis;
public:
    ReduceMeanNode(NodePtr input, size_t axis);
    virtual ~ReduceMeanNode();
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;

    size_t GetAxis() const;
};

#endif
