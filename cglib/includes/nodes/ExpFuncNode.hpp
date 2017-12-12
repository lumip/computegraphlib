#ifndef _EXP_FUNC_NODE_HPP_
#define _EXP_FUNC_NODE_HPP_

#include "nodes/Node.hpp"

class ExpFuncNode : public Node
{
private:
    const ConstNodePtr _input;
public:
    ExpFuncNode(NodePtr input);
    virtual ~ExpFuncNode();
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;
};

#endif
