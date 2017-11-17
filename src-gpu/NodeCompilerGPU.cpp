#include <CL/cl.h>

#include "GraphCompilationGPUStrategy.hpp"
#include "Kernel.hpp"

#include "nodes/InputNode.hpp"
#include "nodes/MatrixMultNode.hpp"
#include "nodes/VariableNode.hpp"
#include "nodes/VectorAddNode.hpp"

extern std::string MatrixMultSource;
extern std::string VectorAddKernelSource;

void GraphCompilationGPUStrategy::CompileInputNode(const InputNode* const node) { }

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

void GraphCompilationGPUStrategy::CompileMatrixMultNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const MatrixMultNode* const node)
{
    const MemoryDimensions inputADims = _dimensionsMap.GetNodeMemoryDimensions(inputANode);
    const MemoryDimensions inputBDims = _dimensionsMap.GetNodeMemoryDimensions(inputBNode);
    const MemoryDimensions resultDims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputABuffer = _bufferMap.at(inputANode);
    const MemoryHandle inputBBuffer = _bufferMap.at(inputBNode);
    const MemoryHandle resultBuffer = _bufferMap.at(node);
    auto m = inputADims.yDim;
    auto n = inputBDims.xDim;
    auto d = inputADims.xDim;
    cl_kernel kernel = CompileKernel(MatrixMultSource);
    _kernels.emplace_back(
           std::unique_ptr<Kernel>(new MatrixMultNodeGPUKernel(kernel,
                                                               _clExecutionQueue,
                                                               inputABuffer,
                                                               inputBBuffer,
                                                               resultBuffer,
                                                               m, n, d))
    );
}

void GraphCompilationGPUStrategy::CompileVariableNode(const VariableNode* const node) { }

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

void GraphCompilationGPUStrategy::CompileVectorAddNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const VectorAddNode* const node)
{
    cl_kernel kernel = CompileKernel(VectorAddKernelSource);
    const MemoryDimensions resultDims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputABuffer = _bufferMap.at(inputANode);
    const MemoryHandle inputBBuffer = _bufferMap.at(inputBNode);
    const MemoryHandle resultBuffer = _bufferMap.at(node);
    _kernels.emplace_back(
           std::unique_ptr<Kernel>(new VectorAddNodeGPUKernel(kernel,
                                                              _clExecutionQueue,
                                                              inputABuffer,
                                                              inputBBuffer,
                                                              resultBuffer,
                                                              resultDims.size()))
    );
}
