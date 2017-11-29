#include "GraphCompilationCPUPlatform.hpp"

#include <cmath>

#include "CompilationMemoryMap.hpp"
#include "nodes/nodes.hpp"
#include "kernels/kernels.hpp"

void GraphCompilationCPUPlatform::CompileConstMultNode(const ConstMultNode* const node)
{
    const MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputBuffer = _bufferMap.at(node->GetInputs()[0]).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
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
    const MemoryHandle inputBuffer = _bufferMap.at(node->GetInputs()[0]).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    _kernels.emplace_back(
        new ExpFuncNodeCPUKernel(inputBuffer, resultBuffer, dims.size())
    );
}

void GraphCompilationCPUPlatform::CompileInputNode(const InputNode* const node) { }

void GraphCompilationCPUPlatform::CompileLogFuncNode(const LogFuncNode* const node)
{
    const MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputBuffer = _bufferMap.at(node->GetInputs()[0]).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    _kernels.emplace_back(
        new LogFuncNodeCPUKernel(inputBuffer, resultBuffer, dims.size())
    );
}

void GraphCompilationCPUPlatform::CompileMatrixMultNode(const MatrixMultNode* const node)
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
           std::unique_ptr<Kernel>(new MatrixMultNodeCPUKernel(inputABuffer,
                                                               inputBBuffer,
                                                               resultBuffer,
                                                               m, n, d))
    );
}

void GraphCompilationCPUPlatform::CompileNegateNode(const NegateNode* const node)
{
    throw std::logic_error("not yet implemented!");
}

void GraphCompilationCPUPlatform::CompileReduceMeanNode(const ReduceMeanNode* const node)
{
    throw std::logic_error("not yet implemented!");
}

void GraphCompilationCPUPlatform::CompileReduceSumNode(const ReduceSumNode* const node)
{
    throw std::logic_error("not yet implemented!");
}

void GraphCompilationCPUPlatform::CompileSliceNode(const SliceNode* const node)
{
    throw std::logic_error("not yet implemented!");
}

void GraphCompilationCPUPlatform::CompileStackNode(const StackNode* const node)
{
    throw std::logic_error("not yet implemented!");
}

void GraphCompilationCPUPlatform::CompileTransposeNode(const TransposeNode* const node)
{
    throw std::logic_error("not yet implemented!");
}

void GraphCompilationCPUPlatform::CompileVariableNode(const VariableNode* const node)
{
    throw std::logic_error("not yet implemented!");
}

void GraphCompilationCPUPlatform::CompileVectorAddNode(const VectorAddNode* const node)
{
    Node::ConstNodeList inputs = node->GetInputs();
    MemoryDimensions resultDims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputABuffer = _bufferMap.at(inputs[0]).get();
    const MemoryHandle inputBBuffer = _bufferMap.at(inputs[1]).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    _kernels.emplace_back(
           std::unique_ptr<Kernel>(new VectorAddNodeCPUKernel(inputABuffer,
                                                              inputBBuffer,
                                                              resultBuffer,
                                                              resultDims.size()))
    );
}

void GraphCompilationCPUPlatform::CompileVectorDivNode(const VectorDivNode* const node)
{
    throw std::logic_error("not yet implemented!");
}

void GraphCompilationCPUPlatform::CompileVectorMultNode(const VectorMultNode* const node)
{
    throw std::logic_error("not yet implemented!");
}
