#ifndef _GRAPH_COMPILATION_GPU_STRATEGY_HPP_
#define _GRAPH_COMPILATION_GPU_STRATEGY_HPP_

#include <memory>
#include <atomic>

#include <CL/cl.h>

#include "types.hpp"
#include "GraphCompilationPlatform.hpp"

#include "kernels/OpenCLCompiler.hpp"
#include "OCLWrappers.hpp"

class CompilationMemoryMap;
class Kernel;

class GraphCompilationGPUPlatform : public GraphCompilationPlatform, public OpenCLCompiler
{
private:
    typedef cl_mem MemoryHandle;
private:
    const OCLWrappers::Context _clContext;
    const cl_device_id _clDevice; // we do not allocate the device so we do not have a wrapper keeping care of releasing it
    const OCLWrappers::Queue _clMemoryQueue;
    const OCLWrappers::Queue _clExecutionQueue;
    std::vector<std::unique_ptr<Kernel>> _kernels;
    std::vector<OCLWrappers::Memory> _memoryBufferLocations;
    std::vector<OCLWrappers::Program> _clPrograms;
    std::map<std::string, OCLWrappers::Kernel> _clKernels;
    OCLWrappers::Event _executionFinishedEvent;
    std::atomic<bool> _isRunning; // used in (asynchronous) OpenCL callback
    std::map<ConstNodePtr, GPUKernel const*> _nodeKernels;
    std::map<std::string, std::pair<OCLWrappers::Memory, float*>> _mappedInputBuffers;
private:
    cl_device_id SelectDevice();
    OCLWrappers::Queue CreateCommandQueue();
    MemoryHandle GetMemoryLocation(const ConstNodePtr node) const;
    GPUKernel const* GetNodeKernel(ConstNodePtr node) const;
    void AllocateMappedMemory(std::string const& inputName, ConstNodePtr const node);
public:
    GraphCompilationGPUPlatform(const CompilationMemoryMap& compilationMemoryMap, OCLWrappers::Context&& context);
    ~GraphCompilationGPUPlatform();
    void AllocateAllMemory();
    void CopyOutputData(const ConstNodePtr outputNode, float* outputBuffer) const;
    void CopyInputData(const ConstNodePtr inputNode, float const* inputBuffer);
    void Evaluate();
    bool IsEvaluating() const;
    void WaitUntilEvaluationFinished() const;
    void WaitUntilDataTransferFinished() const;
    float* GetMappedInputBuffer(std::string const& inputName);

    cl_kernel CompileKernel(const std::string& kernelSource);

    void CompileConstMultNode(const ConstMultNode* const node);
    void CompileExpFuncNode(const ExpFuncNode* const node);
    void CompileInputNode(const InputNode* const node);
    void CompileLogFuncNode(const LogFuncNode* const node);
    void CompileMatrixMultNode(const MatrixMultNode* const node);
    void CompileNegateNode(const NegateNode* const node);
    void CompileReduceMeanNode(const ReduceMeanNode* const node);
    void CompileReduceSumNode(const ReduceSumNode* const node);
    void CompileSliceNode(const SliceNode* const node);
    void CompileStackNode(const StackNode* const node);
    void CompileTransposeNode(const TransposeNode* const node);
    void CompileVariableNode(const VariableNode* const node);
    void CompileVectorAddNode(const VectorAddNode* const node);
    void CompileVectorDivNode(const VectorDivNode* const node);
    void CompileVectorMultNode(const VectorMultNode* const node);
};

#endif
