#ifndef _INPUT_NODE_HPP_
#define _INPUT_NODE_HPP_

#include "Node.hpp"

class InputNode : public Node
{
private:
    std::string _name;
    size_t _dim;
public:
    InputNode(std::string name, size_t dim);
    virtual ~InputNode();
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationContext& context, NodeCompiler& nodeCompiler) const;
    MemoryDimensions GetMemoryDimensions(const InputDimensionsMap& inputDimensions, const std::map<ConstNodePtr, MemoryDimensions>& nodeMemoryDimensions) const;
    std::string ToString() const;
    bool IsInitialized() const;
};

#endif
