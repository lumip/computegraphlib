#include <CL/cl.h>

#include "GraphCompilationGPUPlatform.hpp"
#include "Kernel.hpp"

#include "nodes/InputNode.hpp"
#include "nodes/MatrixMultNode.hpp"
#include "nodes/VariableNode.hpp"
#include "nodes/VectorAddNode.hpp"

extern std::string MatrixMultSource;
extern std::string VectorAddKernelSource;

void GraphCompilationGPUPlatform::CompileInputNode(const InputNode* const node) { }

class MatrixMultNodeGPUKernel : public Kernel
{
private:
    const OCLWrappers::Kernel _kernel;
    const cl_command_queue _queue;
    const cl_mem _inputABuffer;
    const cl_mem _inputBBuffer;
    const cl_mem _resultBuffer;
    const size_t _m;
    const size_t _n;
    const size_t _d;
public:
    MatrixMultNodeGPUKernel(OCLWrappers::Kernel&& kernel, const cl_command_queue queue, cl_mem inputABuffer, cl_mem inputBBuffer, cl_mem resultBuffer, size_t m, size_t n, size_t d)
        : _kernel(std::move(kernel)), _queue(queue), _inputABuffer(inputABuffer), _inputBBuffer(inputBBuffer), _resultBuffer(resultBuffer), _m(m), _n(n), _d(d)
    { }

    ~MatrixMultNodeGPUKernel() { }

    void Run()
    {
        clSetKernelArg(_kernel.get(), 0, sizeof(cl_mem), &_inputABuffer);
        clSetKernelArg(_kernel.get(), 1, sizeof(cl_mem), &_inputBBuffer);
        clSetKernelArg(_kernel.get(), 2, sizeof(cl_mem), &_resultBuffer);
        clSetKernelArg(_kernel.get(), 3, sizeof(cl_uint), &_d);
        size_t globalWorkSize[2] = { _m, _n };
        OCLWrappers::CheckCLError(
                    clEnqueueNDRangeKernel(_queue, _kernel.get(), 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr)
        , "clEnqueueNDRangeKernel");
    }
};

void GraphCompilationGPUPlatform::CompileMatrixMultNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const MatrixMultNode* const node)
{
    const MemoryDimensions inputADims = _dimensionsMap.GetNodeMemoryDimensions(inputANode);
    const MemoryDimensions inputBDims = _dimensionsMap.GetNodeMemoryDimensions(inputBNode);
    const MemoryDimensions resultDims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputABuffer = _bufferMap.at(inputANode).get();
    const MemoryHandle inputBBuffer = _bufferMap.at(inputBNode).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    auto m = inputADims.yDim;
    auto n = inputBDims.xDim;
    auto d = inputADims.xDim;
    OCLWrappers::Kernel kernel = CompileKernel(MatrixMultSource);
    _kernels.emplace_back(
           std::unique_ptr<Kernel>(new MatrixMultNodeGPUKernel(std::move(kernel),
                                                               _clExecutionQueue.get(),
                                                               inputABuffer,
                                                               inputBBuffer,
                                                               resultBuffer,
                                                               m, n, d))
    );
}

void GraphCompilationGPUPlatform::CompileVariableNode(const VariableNode* const node) { }

class VectorAddNodeGPUKernel : public Kernel
{
private:
    const OCLWrappers::Kernel _kernel;
    const cl_command_queue _queue;
    const cl_mem _inputABuffer;
    const cl_mem _inputBBuffer;
    const cl_mem _resultBuffer;
    const size_t _size;
public:
    VectorAddNodeGPUKernel(OCLWrappers::Kernel&& kernel, const cl_command_queue queue, cl_mem inputABuffer, cl_mem inputBBuffer, cl_mem resultBuffer, size_t size)
        : _kernel(std::move(kernel)), _queue(queue), _inputABuffer(inputABuffer), _inputBBuffer(inputBBuffer), _resultBuffer(resultBuffer), _size(size)
    { }

    ~VectorAddNodeGPUKernel() { }

    void Run()
    {
        clSetKernelArg(_kernel.get(), 0, sizeof(cl_mem), &_inputABuffer);
        clSetKernelArg(_kernel.get(), 1, sizeof(cl_mem), &_inputBBuffer);
        clSetKernelArg(_kernel.get(), 2, sizeof(cl_mem), &_resultBuffer);
        size_t globalWorkSize[1] = { _size };
        OCLWrappers::CheckCLError(
            clEnqueueNDRangeKernel(_queue, _kernel.get(), 1, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr)
        , "clEnqueueNDRangeKernel");
    }
};

void GraphCompilationGPUPlatform::CompileVectorAddNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const VectorAddNode* const node)
{
    OCLWrappers::Kernel kernel = CompileKernel(VectorAddKernelSource);
    const MemoryDimensions resultDims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputABuffer = _bufferMap.at(inputANode).get();
    const MemoryHandle inputBBuffer = _bufferMap.at(inputBNode).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    _kernels.emplace_back(
           std::unique_ptr<Kernel>(new VectorAddNodeGPUKernel(std::move(kernel),
                                                              _clExecutionQueue.get(),
                                                              inputABuffer,
                                                              inputBBuffer,
                                                              resultBuffer,
                                                              resultDims.size()))
    );
}
