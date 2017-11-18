#ifndef _GRAPH_COMPILATION_GPU_STRATEGY_HPP_
#define _GRAPH_COMPILATION_GPU_STRATEGY_HPP_

#include <memory>

#include <CL/cl.h>

#include "types.hpp"
#include "GraphCompilationPlatform.hpp"

#include "OCLWrappers.hpp"

class CompilationMemoryMap;
class Kernel;

class GraphCompilationGPUPlatform : public GraphCompilationPlatform
{
private:
    typedef cl_mem MemoryHandle;
    const CompilationMemoryMap& _dimensionsMap;
    const OCLWrappers::Context _clContext;
    const cl_device_id _clDevice; // we do not allocate the device so we do not have a wrapper keeping care of releasing it
    const OCLWrappers::Queue _clMemoryQueue;
    const OCLWrappers::Queue _clExecutionQueue;
    std::vector<std::unique_ptr<Kernel>> _kernels;
    std::map<ConstNodePtr, OCLWrappers::Memory> _bufferMap;
    std::vector<OCLWrappers::Program> _programs;
private:
    cl_device_id SelectDevice();
    OCLWrappers::Queue CreateCommandQueue();
    OCLWrappers::Kernel CompileKernel(const std::string& kernelSource);
public:
    GraphCompilationGPUPlatform(const CompilationMemoryMap& CompilationMemoryMap, OCLWrappers::Context&& context);
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
