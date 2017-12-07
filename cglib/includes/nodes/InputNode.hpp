#ifndef _INPUT_NODE_HPP_
#define _INPUT_NODE_HPP_

#include "Node.hpp"

class InputNode : public Node
{
private:
    const std::string _name;
    const size_t _dim;
public:
    InputNode(std::string name, size_t dim);
    virtual ~InputNode();
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;
};

#endif
