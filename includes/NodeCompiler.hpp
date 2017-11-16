#ifndef _NODE_COMPILER_HPP_
#define _NODE_COMPILER_HPP_

#include <memory>
#include "GraphCompilationContext.hpp"

class Kernel;
class InputNode;
class MatrixMultNode;
class VariableNode;
class VectorAddNode;

class NodeCompiler
{
public:
    virtual ~NodeCompiler() { }
    virtual std::unique_ptr<Kernel> CompileInputNode(const InputNode* const node) = 0;
    virtual std::unique_ptr<Kernel> CompileMatrixMultNode(const NodeMemoryDescriptor inputAMem, const NodeMemoryDescriptor inputBMem, const NodeMemoryDescriptor resultMem) = 0;
    virtual std::unique_ptr<Kernel> CompileVariableNode(const VariableNode* const node) = 0;
    virtual std::unique_ptr<Kernel> CompileVectorAddNode(const NodeMemoryDescriptor inputAMem, const NodeMemoryDescriptor inputBMem, const NodeMemoryDescriptor resultMem) = 0;
};

#endif
