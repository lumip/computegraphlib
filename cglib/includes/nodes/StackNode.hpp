#ifndef _STACK_NODE_HPP_
#define _STACK_NODE_HPP_

#include "nodes/Node.hpp"

class StackNode : public Node
{
private:
    const std::vector<ConstNodePtr> _components;
    const size_t _axis;
public:
    StackNode(const std::vector<NodePtr>&, size_t axis);
    virtual ~StackNode();
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;

    size_t GetAxis() const;
};

#endif
