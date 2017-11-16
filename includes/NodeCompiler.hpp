#ifndef _NODE_COMPILER_HPP_
#define _NODE_COMPILER_HPP_

#include <memory>
#include "types.hpp"

class Kernel;
class InputNode;
class MatrixMultNode;
class VariableNode;
class VectorAddNode;

class NodeCompiler
{
public:
    NodeCompiler() { }
    virtual ~NodeCompiler() { }
    virtual void CompileInputNode(const InputNode* const node) = 0;
    virtual void CompileMatrixMultNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const MatrixMultNode* const node) = 0;
    virtual void CompileVariableNode(const VariableNode* const node) = 0;
    virtual void CompileVectorAddNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const VectorAddNode* const node) = 0;
};

#endif
