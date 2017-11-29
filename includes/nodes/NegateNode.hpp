#ifndef _NEGATE_NODE_HPP_
#define _NEGATE_NODE_HPP_

#include "nodes/Node.hpp"

class NegateNode : public Node
{
private:
    const ConstNodePtr _input;
public:
    NegateNode(NodePtr input);
    virtual ~NegateNode();
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;
    bool IsInitialized() const;
};

#endif
