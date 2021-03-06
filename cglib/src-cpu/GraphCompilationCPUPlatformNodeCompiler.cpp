#include "GraphCompilationCPUPlatform.hpp"

#include <assert.h>

#include "CompilationMemoryMap.hpp"
#include "nodes/nodes.hpp"
#include "kernels/kernels.hpp"

void GraphCompilationCPUPlatform::CompileConstMultNode(const ConstMultNode* const node)
{
    const MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputBuffer = GetMemoryLocation(node->GetInputs()[0]);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    _kernels.emplace_back(
           std::unique_ptr<Kernel>(new ConstMultNodeCPUKernel(inputBuffer,
                                                              resultBuffer,
                                                              node->GetFactor(),
                                                              dims.size()))
    );
}

void GraphCompilationCPUPlatform::CompileExpFuncNode(const ExpFuncNode* const node)
{
    const MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputBuffer = GetMemoryLocation(node->GetInputs()[0]);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    _kernels.emplace_back(
        new ExpFuncNodeCPUKernel(inputBuffer, resultBuffer, dims.size())
    );
}

void GraphCompilationCPUPlatform::CompileInputNode(const InputNode* const node) { }

void GraphCompilationCPUPlatform::CompileLogFuncNode(const LogFuncNode* const node)
{
    const MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputBuffer = GetMemoryLocation(node->GetInputs()[0]);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    _kernels.emplace_back(
        new LogFuncNodeCPUKernel(inputBuffer, resultBuffer, dims.size())
    );
}

void GraphCompilationCPUPlatform::CompileMatrixMultNode(const MatrixMultNode* const node)
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
    _kernels.emplace_back(
           std::unique_ptr<Kernel>(new MatrixMultNodeCPUKernel(inputABuffer,
                                                               inputBBuffer,
                                                               resultBuffer,
                                                               m, n, d))
    );
}

void GraphCompilationCPUPlatform::CompileNegateNode(const NegateNode* const node)
{
    const MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputBuffer = GetMemoryLocation(node->GetInputs()[0]);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    _kernels.emplace_back(
        new NegateNodeCPUKernel(inputBuffer,
                                resultBuffer,
                                dims.size())
    );
}

void GraphCompilationCPUPlatform::CompileReduceMeanNode(const ReduceMeanNode* const node)
{
    ConstNodePtr inputNode = node->GetInputs()[0];
    const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    const MemoryHandle inputBuffer = GetMemoryLocation(inputNode);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    _kernels.emplace_back(
        new ReduceMeanNodeCPUKernel(inputBuffer,
                                    resultBuffer,
                                    inputDims,
                                    node->GetAxis())
    );
}

void GraphCompilationCPUPlatform::CompileReduceSumNode(const ReduceSumNode* const node)
{
    ConstNodePtr inputNode = node->GetInputs()[0];
    const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    const MemoryHandle inputBuffer = GetMemoryLocation(inputNode);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    _kernels.emplace_back(
        new ReduceSumNodeCPUKernel(inputBuffer,
                                   resultBuffer,
                                   inputDims,
                                   node->GetAxis())
    );
}

void GraphCompilationCPUPlatform::CompileSliceNode(const SliceNode* const node)
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
    _kernels.emplace_back(
        new CopyDataCPUKernel(inputBuffer + offset,
                              resultBuffer,
                              inputDims.dims[1-axis],
                              inputStride,
                              1)
    );
}

void GraphCompilationCPUPlatform::CompileStackNode(const StackNode* const node)
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

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        const MemoryHandle inputBuffer = GetMemoryLocation(inputs[i]);
        _kernels.emplace_back(
            new CopyDataCPUKernel(inputBuffer,
                                  resultBuffer + i * offsetStride,
                                  inputDims.size(),
                                  1,
                                  outputStride)
        );
    }
}

void GraphCompilationCPUPlatform::CompileTransposeNode(const TransposeNode* const node)
{
    ConstNodePtr inputNode = node->GetInputs()[0];
    const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    const MemoryHandle inputBuffer = GetMemoryLocation(inputNode);
    const MemoryHandle resultBuffer = GetMemoryLocation(node);
    for (size_t i = 0; i < inputDims.yDim; ++i)
    {
        _kernels.emplace_back(
            new CopyDataCPUKernel(inputBuffer + i * inputDims.xDim,
                                  resultBuffer + i,
                                  inputDims.xDim,
                                  1,
                                  inputDims.yDim)
        );
    }
}

void GraphCompilationCPUPlatform::CompileVariableNode(const VariableNode* const node)
{
    Node::ConstNodeList inputs = node->GetInputs();
    if (inputs.size() > 0)
    {
        ConstNodePtr inputNode = node->GetInputs()[0];
        const MemoryHandle inputBuffer = GetMemoryLocation(inputNode);
        const MemoryHandle resultBuffer = GetMemoryLocation(node);
        if (inputBuffer != resultBuffer) // if the VariableNode has an input path and which doesn't share the buffer, copy data over
        {
            const MemoryDimensions inputDims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
            _kernels.emplace_back(
                new CopyDataCPUKernel(inputBuffer,
                                      resultBuffer,
                                      inputDims.size(),
                                      1, 1)
            );
        }
    }
}

void GraphCompilationCPUPlatform::CompileVectorAddNode(const VectorAddNode* const node)
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
    assert(_dimensionsMap.GetNodeMemoryDimensions(node) == inputADims);
    _kernels.emplace_back(
           new VectorAddNodeCPUKernel(inputABuffer,
                                      inputBBuffer,
                                      resultBuffer,
                                      inputADims,
                                      inputBDims)
    );
}

void GraphCompilationCPUPlatform::CompileVectorDivNode(const VectorDivNode* const node)
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
    _kernels.emplace_back(
        new VectorDivNodeCPUKernel(inputABuffer,
                                   inputBBuffer,
                                   resultBuffer,
                                   inputADims,
                                   inputBDims)
    );
}

void GraphCompilationCPUPlatform::CompileVectorMultNode(const VectorMultNode* const node)
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
            throw std::runtime_error("CompileVectorMultNode: The dimensions of the inputs do not match.");
        }
    }
    assert(_dimensionsMap.GetNodeMemoryDimensions(node) == inputADims);
    _kernels.emplace_back(
           new VectorMultNodeCPUKernel(inputABuffer,
                                      inputBBuffer,
                                      resultBuffer,
                                      inputADims,
                                      inputBDims)
    );
}
