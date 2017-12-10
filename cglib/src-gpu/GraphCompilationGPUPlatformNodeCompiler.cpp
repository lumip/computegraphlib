#include "GraphCompilationGPUPlatform.hpp"

#include <assert.h>
#include <CL/cl.h>

#include "CompilationMemoryMap.hpp"
#include "Kernel.hpp"
#include "nodes/nodes.hpp"

#include "kernels/kernels.hpp"

void GraphCompilationGPUPlatform::CompileConstMultNode(const ConstMultNode* const node)
{
    const MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputBuffer = GetMemoryLocation(node->GetInputs()[0]);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    const GPUKernel::ConstList inputKernels { GetNodeKernel(node->GetInputs()[0]) };
    ConstMultNodeGPUKernel* kernel = new ConstMultNodeGPUKernel(*this,
                                                                _clExecutionQueue.get(),
                                                                inputKernels,
                                                                inputBuffer,
                                                                resultBuffer,
                                                                node->GetFactor(),
                                                                dims.size());
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileExpFuncNode(const ExpFuncNode* const node)
{
    const MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputBuffer = GetMemoryLocation(node->GetInputs()[0]);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    const GPUKernel::ConstList inputKernels { GetNodeKernel(node->GetInputs()[0]) };
    ExpFuncNodeGPUKernel* kernel = new ExpFuncNodeGPUKernel(*this,
                                                            _clExecutionQueue.get(),
                                                            inputKernels,
                                                            inputBuffer,
                                                            resultBuffer,
                                                            dims.size());
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileInputNode(const InputNode* const node)
{
    EmptyGPUKernel* kernel = new EmptyGPUKernel(_clExecutionQueue.get(),
                                                GPUKernel::ConstList());
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileLogFuncNode(const LogFuncNode* const node)
{
    const MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputBuffer = GetMemoryLocation(node->GetInputs()[0]);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    const GPUKernel::ConstList inputKernels { GetNodeKernel(node->GetInputs()[0]) };
    LogFuncNodeGPUKernel* kernel = new LogFuncNodeGPUKernel(*this,
                                                            _clExecutionQueue.get(),
                                                            inputKernels,
                                                            inputBuffer,
                                                            resultBuffer,
                                                            dims.size());
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileMatrixMultNode(const MatrixMultNode* const node)
{
    Node::ConstNodeList inputs = node->GetInputs();
    const MemoryDimensions inputADims = _dimensionsMap.GetNodeMemoryDimensions(inputs[0]);
    const MemoryDimensions inputBDims = _dimensionsMap.GetNodeMemoryDimensions(inputs[1]);
    const MemoryHandle inputABuffer = GetMemoryLocation(inputs[0]);
    const MemoryHandle inputBBuffer = GetMemoryLocation(inputs[1]);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    auto m = inputADims.yDim;
    auto n = inputBDims.xDim;
    auto d = inputADims.xDim;
    const GPUKernel::ConstList inputKernels { GetNodeKernel(inputs[0]), GetNodeKernel(inputs[1]) };
    MatrixMultNodeGPUKernel* kernel = new MatrixMultNodeGPUKernel(*this,
                                                                  _clExecutionQueue.get(),
                                                                  inputKernels,
                                                                  inputABuffer,
                                                                  inputBBuffer,
                                                                  resultBuffer,
                                                                  m, n, d);
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileNegateNode(const NegateNode* const node)
{
    const MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputBuffer = GetMemoryLocation(node->GetInputs()[0]);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    const GPUKernel::ConstList inputKernels { GetNodeKernel(node->GetInputs()[0]) };
    NegateNodeGPUKernel* kernel = new NegateNodeGPUKernel(*this,
                                                          _clExecutionQueue.get(),
                                                          inputKernels,
                                                          inputBuffer,
                                                          resultBuffer,
                                                          dims.size());
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileReduceMeanNode(const ReduceMeanNode* const node)
{
    ConstNodePtr inputNode = node->GetInputs()[0];
    const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    const MemoryHandle inputBuffer = GetMemoryLocation(inputNode);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    const GPUKernel::ConstList inputKernels { GetNodeKernel(inputNode) };
    ReduceMeanNodeGPUKernel* kernel = new ReduceMeanNodeGPUKernel(*this,
                                                                  _clExecutionQueue.get(),
                                                                  inputKernels,
                                                                  inputBuffer,
                                                                  resultBuffer,
                                                                  inputDims,
                                                                  node->GetAxis());
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileReduceSumNode(const ReduceSumNode* const node)
{
    ConstNodePtr inputNode = node->GetInputs()[0];
    const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    const MemoryHandle inputBuffer = GetMemoryLocation(inputNode);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    const GPUKernel::ConstList inputKernels { GetNodeKernel(inputNode) };
    ReduceSumNodeGPUKernel* kernel = new ReduceSumNodeGPUKernel(*this,
                                                                _clExecutionQueue.get(),
                                                                inputKernels,
                                                                inputBuffer,
                                                                resultBuffer,
                                                                inputDims,
                                                                node->GetAxis());
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileSliceNode(const SliceNode* const node)
{
    ConstNodePtr inputNode = node->GetInputs()[0];
    const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    const MemoryHandle inputBuffer = GetMemoryLocation(inputNode);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    const size_t axis = node->GetAxis();
    const size_t sliceId = node->GetSliceId();
    size_t inputStride = 1;
    if (axis == 1)
    {
        inputStride = inputDims.xDim;
    }
    size_t offset = sliceId;
    if (axis == 0)
    {
        offset *= inputDims.xDim;
    }
    const GPUKernel::ConstList inputKernels { GetNodeKernel(inputNode) };
    CopyDataGPUKernel* kernel = new CopyDataGPUKernel(*this,
                                                      _clExecutionQueue.get(),
                                                      inputKernels,
                                                      inputBuffer,
                                                      resultBuffer,
                                                      inputDims.dims[1-axis],   // count
                                                      offset, 0,                // offset
                                                      inputStride, 1);          // strides
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileStackNode(const StackNode* const node)
{
    Node::ConstNodeList inputs = node->GetInputs();
    const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputs[0]);
    assert(inputDims.xDim == 1 || inputDims.yDim == 1);
    const MemoryDimensions resultDims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    const size_t axis = node->GetAxis();

    size_t outputStride = 1;
    if (inputDims.xDim == 1)
    {
        outputStride = resultDims.xDim;
    }

    size_t offsetStride = inputDims.size();
    if (axis == 1)
    {
        offsetStride = inputDims.xDim;
    }

    GPUKernel::ConstList processKernels;
    processKernels.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        const MemoryHandle inputBuffer = GetMemoryLocation(inputs[i]);
        const GPUKernel::ConstList inputKernels { GetNodeKernel(inputs[i]) };
        CopyDataGPUKernel* kernel = new CopyDataGPUKernel(*this,
                                                          _clExecutionQueue.get(),
                                                          inputKernels,
                                                          inputBuffer,
                                                          resultBuffer,
                                                          inputDims.size(),     // count
                                                          0, i * offsetStride,  // offset
                                                          1, outputStride);     // strides
        processKernels.push_back(kernel);
        _kernels.emplace_back(kernel);
    }
    WaitForGPUKernel* kernel = new WaitForGPUKernel(*this,
                                                    _clExecutionQueue.get(),
                                                    processKernels);
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileTransposeNode(const TransposeNode* const node)
{
    ConstNodePtr inputNode = node->GetInputs()[0];
    const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    const MemoryHandle inputBuffer = GetMemoryLocation(inputNode);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    const GPUKernel::ConstList inputKernels { GetNodeKernel(inputNode) };
    GPUKernel::ConstList processKernels;
    processKernels.reserve(inputDims.yDim);
    for (size_t i = 0; i < inputDims.yDim; ++i)
    {
        CopyDataGPUKernel* kernel = new CopyDataGPUKernel(*this,
                                                          _clExecutionQueue.get(),
                                                          inputKernels,
                                                          inputBuffer,
                                                          resultBuffer,
                                                          inputDims.xDim,           // count
                                                          i * inputDims.xDim, i,    // offset
                                                          1, inputDims.yDim);       // strides
        processKernels.push_back(kernel);
        _kernels.emplace_back(kernel);
    }
    WaitForGPUKernel* kernel = new WaitForGPUKernel(*this,
                                                    _clExecutionQueue.get(),
                                                    processKernels);
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileVariableNode(const VariableNode* const node)
{
    GPUKernel* kernel = nullptr;
    Node::ConstNodeList inputs = node->GetInputs();
    GPUKernel::ConstList inputKernels; // deliberately empty in all cases (we do not want to wait for a computation that will occur after us in the pipeline!)
    if (inputs.size() > 0)
    {
        ConstNodePtr inputNode = node->GetInputs()[0];
        const MemoryHandle inputBuffer = GetMemoryLocation(inputNode);
        const MemoryHandle resultBuffer = GetMemoryLocation(node);
        if (inputBuffer != resultBuffer) // if the VariableNode has an input path and which doesn't share the buffer, copy data over
        {
            const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
             kernel = new CopyDataGPUKernel(*this,
                                            _clExecutionQueue.get(),
                                            inputKernels,
                                            inputBuffer,
                                            resultBuffer,
                                            inputDims.size(),
                                            0, 0, 1, 1);
        }
    }
    if (kernel == nullptr)
    {
        kernel = new EmptyGPUKernel(_clExecutionQueue.get(),
                                    inputKernels);
    }
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileVectorAddNode(const VectorAddNode* const node)
{
    Node::ConstNodeList inputs = node->GetInputs();
    MemoryDimensions inputADims = _dimensionsMap.GetNodeMemoryDimensions(inputs[0]);
    MemoryDimensions inputBDims = _dimensionsMap.GetNodeMemoryDimensions(inputs[1]);
    MemoryHandle inputABuffer = GetMemoryLocation(inputs[0]);
    MemoryHandle inputBBuffer = GetMemoryLocation(inputs[1]);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    if (inputADims != inputBDims)
    {
        if (((inputADims.yDim == inputBDims.yDim) && (inputBDims.xDim % inputADims.xDim == 0)) ||
            ((inputADims.xDim == inputBDims.xDim) && (inputBDims.yDim % inputADims.yDim == 0)))
        {
            std::swap(inputADims, inputBDims);
            std::swap(inputABuffer, inputBBuffer);
        }
        else if (!(((inputADims.yDim == inputBDims.yDim) && (inputADims.xDim % inputBDims.xDim == 0)) ||
                   ((inputADims.xDim == inputBDims.xDim) && (inputADims.yDim % inputBDims.yDim == 0))))
        {
            throw std::runtime_error("CompileVectorAddNode: The dimensions of the inputs do not match.");
        }
    }
    const GPUKernel::ConstList inputKernels { GetNodeKernel(inputs[0]), GetNodeKernel(inputs[1]) };
    assert(_dimensionsMap.GetNodeMemoryDimensions(node) == inputADims);
    VectorAddNodeGPUKernel* kernel =  new VectorAddNodeGPUKernel(*this,
                                                                 _clExecutionQueue.get(),
                                                                 inputKernels,
                                                                 inputABuffer,
                                                                 inputBBuffer,
                                                                 resultBuffer,
                                                                 inputADims,
                                                                 inputBDims);
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileVectorDivNode(const VectorDivNode* const node)
{
    Node::ConstNodeList inputs = node->GetInputs();
    MemoryDimensions inputADims = _dimensionsMap.GetNodeMemoryDimensions(inputs[0]);
    MemoryDimensions inputBDims = _dimensionsMap.GetNodeMemoryDimensions(inputs[1]);
    const MemoryHandle inputABuffer = GetMemoryLocation(inputs[0]);
    const MemoryHandle inputBBuffer = GetMemoryLocation(inputs[1]);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    assert(((inputADims.yDim == inputBDims.yDim) && (inputADims.xDim % inputBDims.xDim == 0)) ||
           ((inputADims.xDim == inputBDims.xDim) && (inputADims.yDim % inputBDims.yDim == 0)));
    assert(_dimensionsMap.GetNodeMemoryDimensions(node) == inputADims);
    const GPUKernel::ConstList inputKernels { GetNodeKernel(inputs[0]), GetNodeKernel(inputs[1]) };
    VectorDivNodeGPUKernel* kernel = new VectorDivNodeGPUKernel(*this,
                                                                _clExecutionQueue.get(),
                                                                inputKernels,
                                                                inputABuffer,
                                                                inputBBuffer,
                                                                resultBuffer,
                                                                inputADims,
                                                                inputBDims);
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}

void GraphCompilationGPUPlatform::CompileVectorMultNode(const VectorMultNode* const node)
{
    Node::ConstNodeList inputs = node->GetInputs();
    MemoryDimensions inputADims = _dimensionsMap.GetNodeMemoryDimensions(inputs[0]);
    MemoryDimensions inputBDims = _dimensionsMap.GetNodeMemoryDimensions(inputs[1]);
    MemoryHandle inputABuffer = GetMemoryLocation(inputs[0]);
    MemoryHandle inputBBuffer = GetMemoryLocation(inputs[1]);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    if (inputADims != inputBDims)
    {
        if (((inputADims.yDim == inputBDims.yDim) && (inputBDims.xDim % inputADims.xDim == 0)) ||
            ((inputADims.xDim == inputBDims.xDim) && (inputBDims.yDim % inputADims.yDim == 0)))
        {
            std::swap(inputADims, inputBDims);
            std::swap(inputABuffer, inputBBuffer);
        }
        else if (!(((inputADims.yDim == inputBDims.yDim) && (inputADims.xDim % inputBDims.xDim == 0)) ||
                   ((inputADims.xDim == inputBDims.xDim) && (inputADims.yDim % inputBDims.yDim == 0))))
        {
            throw std::runtime_error("CompileVectorAddNode: The dimensions of the inputs do not match.");
        }
    }
    const GPUKernel::ConstList inputKernels { GetNodeKernel(inputs[0]), GetNodeKernel(inputs[1]) };
    assert(_dimensionsMap.GetNodeMemoryDimensions(node) == inputADims);
    VectorMultNodeGPUKernel* kernel = new VectorMultNodeGPUKernel(*this,
                                                                  _clExecutionQueue.get(),
                                                                  inputKernels,
                                                                  inputABuffer,
                                                                  inputBBuffer,
                                                                  resultBuffer,
                                                                  inputADims,
                                                                  inputBDims);
    _nodeKernels[node] = kernel;
    _kernels.emplace_back(kernel);
}
