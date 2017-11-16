#ifndef _NODE_COMPILER_CPU_HPP_
#define _NODE_COMPILER_CPU_HPP_

#include "NodeCompiler.hpp"

class NodeCompilerCPU : public NodeCompiler
{
public:
    NodeCompilerCPU();
    ~NodeCompilerCPU();
    std::unique_ptr<Kernel> CompileInputNode(const InputNode* const node);
    std::unique_ptr<Kernel> CompileMatrixMultNode(const NodeMemoryDescriptor inputAMem, const NodeMemoryDescriptor inputBMem, const NodeMemoryDescriptor resultMem);
    std::unique_ptr<Kernel> CompileVariableNode(const VariableNode* const node);
    std::unique_ptr<Kernel> CompileVectorAddNode(const NodeMemoryDescriptor inputAMem, const NodeMemoryDescriptor inputBMem, const NodeMemoryDescriptor resultMem);
};

#endif
