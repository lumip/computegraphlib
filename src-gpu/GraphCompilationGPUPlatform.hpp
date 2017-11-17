#ifndef _GRAPH_COMPILATION_GPU_STRATEGY_HPP_
#define _GRAPH_COMPILATION_GPU_STRATEGY_HPP_

#include <CL/cl.h>

#include "CompilationMemoryMap.hpp"

class CompilationMemoryMap;

class GraphCompilationGPUPlatform : public GraphCompilationPlatform
{
private:
    typedef cl_mem MemoryHandle;
    const CompilationMemoryMap& _dimensionsMap;
    const cl_context _clContext;
    const cl_device_id _clDevice;
    const cl_command_queue _clMemoryQueue;
    const cl_command_queue _clExecutionQueue;
    std::vector<std::unique_ptr<Kernel>> _kernels;
    std::map<ConstNodePtr, MemoryHandle> _bufferMap;
private:
    cl_device_id SelectDevice();
    cl_command_queue CreateCommandQueue();
    cl_kernel CompileKernel(const std::string& kernelSource);
public:
    static void CheckCLError(cl_int status);
    static void CheckCLError(cl_int status, const std::string& methodName);
public:
    GraphCompilationGPUPlatform(const CompilationMemoryMap& CompilationMemoryMap, cl_context context);
    ~GraphCompilationGPUPlatform();
    void AllocateMemory(const ConstNodePtr node);
    void CopyOutputData(const ConstNodePtr outputNode, DataBuffer& outputBuffer) const;
    void CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer);
    void Evaluate();

    void CompileInputNode(const InputNode* const node);
    void CompileMatrixMultNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const MatrixMultNode* const node);
    void CompileVariableNode(const VariableNode* const node);
    void CompileVectorAddNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const VectorAddNode* const node);
};

#endif
