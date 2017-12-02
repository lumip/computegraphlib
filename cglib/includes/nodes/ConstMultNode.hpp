#ifndef _CONST_MULT_NODE_HPP_
#define _CONST_MULT_NODE_HPP_

#include "nodes/Node.hpp"

class ConstMultNode : public Node
{
private:
    const ConstNodePtr _input;
    const float _factor;
public:
    ConstMultNode(NodePtr input, float constantFactor);
    virtual ~ConstMultNode();
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;
    bool IsInitialized() const;

    float GetFactor() const;
};

#endif
