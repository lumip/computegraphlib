#ifndef _LOG_FUNC_NODE_HPP_
#define _LOG_FUNC_NODE_HPP_

#include "nodes/Node.hpp"

class LogFuncNode : public Node
{
private:
    const ConstNodePtr _input;
public:
    LogFuncNode(NodePtr input);
    virtual ~LogFuncNode();
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;
};

#endif
