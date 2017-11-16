#include "CL/cl.h"
#include "NodeCompilerGPU.hpp"
#include "GraphCompilationGPUStrategy.hpp"
#include "Kernel.hpp"

#include "nodes/InputNode.hpp"
#include "nodes/MatrixMultNode.hpp"
#include "nodes/VariableNode.hpp"
#include "nodes/VectorAddNode.hpp"

extern std::string MatrixMultSource;
extern std::string VectorAddKernelSource;

NodeCompilerGPU::NodeCompilerGPU(GraphCompilationGPUStrategy* const gpuStrat)
    : _strategy(gpuStrat)
{

}

NodeCompilerGPU::~NodeCompilerGPU() { }

std::unique_ptr<Kernel> NodeCompilerGPU::CompileInputNode(const InputNode * const node)
{

}

class MatrixMultNodeGPUKernel : public Kernel
{
private:
    const cl_kernel _kernel;
    const cl_command_queue _queue;
    const cl_mem _inputABuffer;
    const cl_mem _inputBBuffer;
    const cl_mem _resultBuffer;
    const size_t _m;
    const size_t _n;
    const size_t _d;
public:
    MatrixMultNodeGPUKernel(const cl_kernel kernel, const cl_command_queue queue, cl_mem inputABuffer, cl_mem inputBBuffer, cl_mem resultBuffer, size_t m, size_t n, size_t d)
        : _kernel(kernel), _queue(queue), _inputABuffer(inputABuffer), _inputBBuffer(inputBBuffer), _resultBuffer(resultBuffer), _m(m), _n(n), _d(d)
    {

    }
    ~MatrixMultNodeGPUKernel() { }
    void Run()
    {
        clSetKernelArg(_kernel, 0, sizeof(cl_mem), &_inputABuffer);
        clSetKernelArg(_kernel, 1, sizeof(cl_mem), &_inputBBuffer);
        clSetKernelArg(_kernel, 2, sizeof(cl_mem), &_resultBuffer);
        clSetKernelArg(_kernel, 3, sizeof(cl_uint), &_d);
        size_t globalWorkSize[2] = { _m, _n };
        GraphCompilationGPUStrategy::CheckCLError(
                    clEnqueueNDRangeKernel(_queue, _kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr)
        , "clEnqueueNDRangeKernel");
    }
};

std::unique_ptr<Kernel> NodeCompilerGPU::CompileMatrixMultNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const MatrixMultNode* const node)
{
    cl_kernel kernel = _strategy->CompileKernel(MatrixMultSource);
    return std::unique_ptr<Kernel>(new MatrixMultNodeGPUKernel(kernel,
                                                               _strategy->_clExecutionQueue,
                                                               reinterpret_cast<const cl_mem>(inputAMem.handle),
                                                               reinterpret_cast<const cl_mem>(inputBMem.handle),
                                                               reinterpret_cast<const cl_mem>(resultMem.handle),
                                                               inputAMem.dimensions.yDim,
                                                               inputBMem.dimensions.xDim,
                                                               inputAMem.dimensions.xDim));
}

std::unique_ptr<Kernel> NodeCompilerGPU::CompileVariableNode(const VariableNode * const node)
{

}

class VectorAddNodeGPUKernel : public Kernel
{
private:
    const cl_kernel _kernel;
    const cl_command_queue _queue;
    const cl_mem _inputABuffer;
    const cl_mem _inputBBuffer;
    const cl_mem _resultBuffer;
    const size_t _size;
public:
    VectorAddNodeGPUKernel(const cl_kernel kernel, const cl_command_queue queue, cl_mem inputABuffer, cl_mem inputBBuffer, cl_mem resultBuffer, size_t size)
        : _kernel(kernel), _queue(queue), _inputABuffer(inputABuffer), _inputBBuffer(inputBBuffer), _resultBuffer(resultBuffer), _size(size) { }
    ~VectorAddNodeGPUKernel() { }
    void Run()
    {
        clSetKernelArg(_kernel, 0, sizeof(cl_mem), &_inputABuffer);
        clSetKernelArg(_kernel, 1, sizeof(cl_mem), &_inputBBuffer);
        clSetKernelArg(_kernel, 2, sizeof(cl_mem), &_resultBuffer);
        size_t globalWorkSize[1] = { _size };
        clEnqueueNDRangeKernel(_queue, _kernel, 1, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    }
};

std::unique_ptr<Kernel> NodeCompilerGPU::CompileVectorAddNode(const NodeMemoryDescriptor inputAMem, const NodeMemoryDescriptor inputBMem, const NodeMemoryDescriptor resultMem)
{
    cl_kernel kernel = _strategy->CompileKernel(VectorAddKernelSource);
    return std::unique_ptr<Kernel>(new VectorAddNodeGPUKernel(kernel,
                                                              _strategy->_clExecutionQueue,
                                                              reinterpret_cast<const cl_mem>(inputAMem.handle),
                                                              reinterpret_cast<const cl_mem>(inputBMem.handle),
                                                              reinterpret_cast<const cl_mem>(resultMem.handle),
                                                              resultMem.dimensions.size()));
}
