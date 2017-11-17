/*#ifndef _NODE_COMPILER_GPU_HPP_
#define _NODE_COMPILER_GPU_HPP_

#include "NodeCompiler.hpp"
#include "GraphCompilationGPUStrategy.hpp"

class NodeCompilerGPU : public NodeCompiler
{
private:
    GraphCompilationGPUStrategy* const _strategy;
public:
    NodeCompilerGPU(GraphCompilationGPUStrategy* const gpuStrat);
    ~NodeCompilerGPU();
    std::unique_ptr<Kernel> CompileInputNode(const InputNode* const node);
    std::unique_ptr<Kernel> CompileMatrixMultNode(const NodeMemoryDescriptor inputAMem, const NodeMemoryDescriptor inputBMem, const NodeMemoryDescriptor resultMem);
    std::unique_ptr<Kernel> CompileVariableNode(const VariableNode* const node);
    std::unique_ptr<Kernel> CompileVectorAddNode(const NodeMemoryDescriptor inputAMem, const NodeMemoryDescriptor inputBMem, const NodeMemoryDescriptor resultMem);
};

#endif*/
