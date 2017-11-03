#ifndef _VARIABLE_NODE_HPP_
#define _VARIABLE_NODE_HPP_

#include "nodes/Node.hpp"

class VariableNode : public Node
{
private:
    std::string _name;
    size_t _dim;
    NodePtr _input;
public:
    VariableNode(std::string name, size_t dim);
    virtual ~VariableNode();
    void SetInput(const NodePtr inputNode);
    ConstNodeMap GetInputs() const;
    std::unique_ptr<const Kernel> Compile(GraphCompilationContext* const context) const;
    std::string ToString() const;
};

#endif
