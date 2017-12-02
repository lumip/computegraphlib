#ifndef _VARIABLE_NODE_HPP_
#define _VARIABLE_NODE_HPP_

#include "nodes/Node.hpp"

class VariableNode : public Node
{
private:
    const std::string _name;
    const size_t _xDim;
    const size_t _yDim;
    NodePtr _input;
public:
    VariableNode(std::string name, size_t xDim, size_t yDim);
    virtual ~VariableNode();
    void SetInput(const NodePtr inputNode);
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;
    bool IsInitialized() const;
};

#endif
