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
    const MemoryHandle inputBuffer = _bufferMap.at(node->GetInputs()[0]).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    _kernels.emplace_back(
        new ConstMultNodeGPUKernel(*this,
                                   _clExecutionQueue.get(),
                                   inputBuffer,
                                   resultBuffer,
                                   node->GetFactor(),
                                   dims.size())
    );
}

void GraphCompilationGPUPlatform::CompileExpFuncNode(const ExpFuncNode* const node)
{
    const MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputBuffer = _bufferMap.at(node->GetInputs()[0]).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    _kernels.emplace_back(
        new ExpFuncNodeGPUKernel(*this,
                                 _clExecutionQueue.get(),
                                 inputBuffer,
                                 resultBuffer,
                                 dims.size())
    );
}

void GraphCompilationGPUPlatform::CompileInputNode(const InputNode* const node) { }

void GraphCompilationGPUPlatform::CompileLogFuncNode(const LogFuncNode* const node)
{
    const MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputBuffer = _bufferMap.at(node->GetInputs()[0]).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    _kernels.emplace_back(
        new LogFuncNodeGPUKernel(*this,
                                 _clExecutionQueue.get(),
                                 inputBuffer,
                                 resultBuffer,
                                 dims.size())
    );
}

void GraphCompilationGPUPlatform::CompileMatrixMultNode(const MatrixMultNode* const node)
{
    Node::ConstNodeList inputs = node->GetInputs();
    const MemoryDimensions inputADims = _dimensionsMap.GetNodeMemoryDimensions(inputs[0]);
    const MemoryDimensions inputBDims = _dimensionsMap.GetNodeMemoryDimensions(inputs[1]);
    const MemoryHandle inputABuffer = _bufferMap.at(inputs[0]).get();
    const MemoryHandle inputBBuffer = _bufferMap.at(inputs[1]).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    auto m = inputADims.yDim;
    auto n = inputBDims.xDim;
    auto d = inputADims.xDim;
    _kernels.emplace_back(
           new MatrixMultNodeGPUKernel(*this,
                                       _clExecutionQueue.get(),
                                       inputABuffer,
                                       inputBBuffer,
                                       resultBuffer,
                                       m, n, d)
    );
}

void GraphCompilationGPUPlatform::CompileNegateNode(const NegateNode* const node)
{
    const MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputBuffer = _bufferMap.at(node->GetInputs()[0]).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    _kernels.emplace_back(
        new NegateNodeGPUKernel(*this,
                                _clExecutionQueue.get(),
                                inputBuffer,
                                resultBuffer,
                                dims.size())
    );
}

void GraphCompilationGPUPlatform::CompileReduceMeanNode(const ReduceMeanNode* const node)
{
    ConstNodePtr inputNode = node->GetInputs()[0];
    const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    const MemoryHandle inputBuffer = _bufferMap.at(inputNode).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    _kernels.emplace_back(
        new ReduceMeanNodeGPUKernel(*this,
                                    _clExecutionQueue.get(),
                                    inputBuffer,
                                    resultBuffer,
                                    inputDims,
                                    node->GetAxis())
    );
}

void GraphCompilationGPUPlatform::CompileReduceSumNode(const ReduceSumNode* const node)
{
    ConstNodePtr inputNode = node->GetInputs()[0];
    const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    const MemoryHandle inputBuffer = _bufferMap.at(inputNode).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    _kernels.emplace_back(
        new ReduceSumNodeGPUKernel(*this,
                                   _clExecutionQueue.get(),
                                   inputBuffer,
                                   resultBuffer,
                                   inputDims,
                                   node->GetAxis())
    );
}

void GraphCompilationGPUPlatform::CompileSliceNode(const SliceNode* const node)
{
    ConstNodePtr inputNode = node->GetInputs()[0];
    const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    const MemoryHandle inputBuffer = _bufferMap.at(inputNode).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
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
    _kernels.emplace_back(
        new CopyDataGPUKernel(*this,
                              _clExecutionQueue.get(),
                              inputBuffer,
                              resultBuffer,
                              inputDims.dims[1-axis],   // count
                              offset, 0,                // offset
                              inputStride, 1)           // strides
    );
}

void GraphCompilationGPUPlatform::CompileStackNode(const StackNode* const node)
{
    Node::ConstNodeList inputs = node->GetInputs();
    const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputs[0]);
    assert(inputDims.xDim == 1 || inputDims.yDim == 1);
    const MemoryDimensions resultDims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
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

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        const MemoryHandle inputBuffer = _bufferMap.at(inputs[i]).get();
        _kernels.emplace_back(
            new CopyDataGPUKernel(*this,            // todo: this will result in compiling and storing i copies of the kernel. should this be avoided?
                                  _clExecutionQueue.get(),
                                  inputBuffer,
                                  resultBuffer,
                                  inputDims.size(),     // count
                                  0, i * offsetStride,  // offset
                                  1, outputStride)      // strides
        );
    }
}

void GraphCompilationGPUPlatform::CompileTransposeNode(const TransposeNode* const node)
{
    ConstNodePtr inputNode = node->GetInputs()[0];
    const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    const MemoryHandle inputBuffer = _bufferMap.at(inputNode).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    for (size_t i = 0; i < inputDims.yDim; ++i)
    {
        _kernels.emplace_back(
            new CopyDataGPUKernel(*this,            // todo: this will result in compiling and storing i copies of the kernel. should this be avoided?
                                  _clExecutionQueue.get(),
                                  inputBuffer,
                                  resultBuffer,
                                  inputDims.xDim,           // count
                                  i * inputDims.xDim, i,    // offset
                                  1, inputDims.yDim)        // strides
        );
    }
}

void GraphCompilationGPUPlatform::CompileVariableNode(const VariableNode* const node) { }

void GraphCompilationGPUPlatform::CompileVectorAddNode(const VectorAddNode* const node)
{
    Node::ConstNodeList inputs = node->GetInputs();
    MemoryDimensions inputADims = _dimensionsMap.GetNodeMemoryDimensions(inputs[0]);
    MemoryDimensions inputBDims = _dimensionsMap.GetNodeMemoryDimensions(inputs[1]);
    MemoryHandle inputABuffer = _bufferMap.at(inputs[0]).get();
    MemoryHandle inputBBuffer = _bufferMap.at(inputs[1]).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
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
    assert(_dimensionsMap.GetNodeMemoryDimensions(node) == inputADims);
    _kernels.emplace_back(
           new VectorAddNodeGPUKernel(*this,
                                      _clExecutionQueue.get(),
                                      inputABuffer,
                                      inputBBuffer,
                                      resultBuffer,
                                      inputADims,
                                      inputBDims)
    );
}

void GraphCompilationGPUPlatform::CompileVectorDivNode(const VectorDivNode* const node)
{
    Node::ConstNodeList inputs = node->GetInputs();
    MemoryDimensions inputADims = _dimensionsMap.GetNodeMemoryDimensions(inputs[0]);
    MemoryDimensions inputBDims = _dimensionsMap.GetNodeMemoryDimensions(inputs[1]);
    const MemoryHandle inputABuffer = _bufferMap.at(inputs[0]).get();
    const MemoryHandle inputBBuffer = _bufferMap.at(inputs[1]).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    assert(((inputADims.yDim == inputBDims.yDim) && (inputADims.xDim % inputBDims.xDim == 0)) ||
           ((inputADims.xDim == inputBDims.xDim) && (inputADims.yDim % inputBDims.yDim == 0)));
    assert(_dimensionsMap.GetNodeMemoryDimensions(node) == inputADims);
    _kernels.emplace_back(
        new VectorDivNodeGPUKernel(*this,
                                   _clExecutionQueue.get(),
                                   inputABuffer,
                                   inputBBuffer,
                                   resultBuffer,
                                   inputADims,
                                   inputBDims)
    );
}

void GraphCompilationGPUPlatform::CompileVectorMultNode(const VectorMultNode* const node)
{
    Node::ConstNodeList inputs = node->GetInputs();
    MemoryDimensions inputADims = _dimensionsMap.GetNodeMemoryDimensions(inputs[0]);
    MemoryDimensions inputBDims = _dimensionsMap.GetNodeMemoryDimensions(inputs[1]);
    MemoryHandle inputABuffer = _bufferMap.at(inputs[0]).get();
    MemoryHandle inputBBuffer = _bufferMap.at(inputs[1]).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
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
    assert(_dimensionsMap.GetNodeMemoryDimensions(node) == inputADims);
    _kernels.emplace_back(
           new VectorMultNodeGPUKernel(*this,
                                       _clExecutionQueue.get(),
                                       inputABuffer,
                                       inputBBuffer,
                                       resultBuffer,
                                       inputADims,
                                       inputBDims)
    );
}
