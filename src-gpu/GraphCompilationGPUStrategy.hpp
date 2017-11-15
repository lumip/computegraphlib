#ifdef GPU

#ifndef _GRAPH_COMPILATION_GPU_STRATEGY_HPP_
#define _GRAPH_COMPILATION_GPU_STRATEGY_HPP_

#include <CL/cl.h>

#include "GraphCompilationContext.hpp"

class GraphCompilationGPUStrategy : public GraphCompilationTargetStrategy
{
private:
    const cl_context _clContext;
    const cl_command_queue _clMemoryQueue;
    const cl_command_queue _clExecutionQueue;
    std::vector<std::unique_ptr<Kernel>> _kernels;
private:
    cl_command_queue CreateCommandQueue();
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
    cl_program CompileKernel(const std::string& kernelSource);

};


#endif

#endif

