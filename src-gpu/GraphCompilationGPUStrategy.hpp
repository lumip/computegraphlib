#ifndef _GRAPH_COMPILATION_GPU_STRATEGY_HPP_
#define _GRAPH_COMPILATION_GPU_STRATEGY_HPP_

#include <CL/cl.h>

#include "GraphCompilationContext.hpp"

class GraphCompilationGPUStrategy : public GraphCompilationPlatform
{
private:
    const cl_context _clContext;
    const cl_device_id _clDevice;
    const cl_command_queue _clMemoryQueue;
    std::vector<std::unique_ptr<Kernel>> _kernels;
public:
    const cl_command_queue _clExecutionQueue;
private:
    cl_device_id SelectDevice();
    cl_command_queue CreateCommandQueue();
    cl_kernel CompileKernel(const std::string& kernelSource);
public:
    static void CheckCLError(cl_int status);
    static void CheckCLError(cl_int status, const std::string& methodName);
public:
    GraphCompilationGPUStrategy(cl_context context);
    ~GraphCompilationGPUStrategy();
    NodeMemoryHandle AllocateMemory(size_t size);
    void DeallocateMemory(const NodeMemoryHandle mem);
    void EnqueueKernel(std::unique_ptr<Kernel>&& kernel);
    void CopyOutputData(const NodeMemoryHandle outputNodeMemory, DataBuffer& outputBuffer, size_t size) const;
    void CopyInputData(const NodeMemoryHandle inputNodeMemory, InputDataBuffer& inputBuffer, size_t size);
    void Evaluate(const std::vector<std::pair<const NodeMemoryDescriptor, InputDataBuffer&>>& inputData);

    void CompileInputNode(const InputNode* const node);
    void CompileMatrixMultNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const MatrixMultNode* const node);
    void CompileVariableNode(const VariableNode* const node);
    void CompileVectorAddNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const VectorAddNode* const node);
};

#endif
