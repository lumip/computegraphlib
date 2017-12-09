#ifndef _VARIABLE_NODE_HPP_
#define _VARIABLE_NODE_HPP_

#include "nodes/Node.hpp"

class VariableNode : public Node
{
private:
    const std::string _name;
    NodePtr _input;
public:
    VariableNode(std::string name);
    virtual ~VariableNode();
    void SetInput(const NodePtr inputNode);
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;
};

#endif
