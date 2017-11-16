/*
#ifndef _NODE_COMPILER_CPU_HPP_
#define _NODE_COMPILER_CPU_HPP_

#include "NodeCompiler.hpp"

class NodeCompilerCPU : public NodeCompiler
{
public:
    NodeCompilerCPU();
    ~NodeCompilerCPU();
    void CompileInputNode(const InputNode* const node);
    void CompileMatrixMultNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const MatrixMultNode* const node);
    void CompileVariableNode(const VariableNode* const node);
    void CompileVectorAddNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const VectorAddNode* const node);
};

#endif
*/
