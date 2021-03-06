#include "VectorMultNodeGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"
#include "../OCLWrappers.hpp"

const std::string VectorMultNodeGPUKernel::KernelSource = R"==kernel==(
uint getIndex(uint const y, uint const x, uint const stride)
{
    return y * stride + x;
}

__kernel void main(__global float const * const vecA, __global float const * const vecB, __global float* const vecResult, uint const sizeAx, uint const sizeBy, uint const sizeBx, uint const maxId)
{
    // get id into vector A -- the bigger one
    uint idA = get_global_id(0);
    if (idA >= maxId) return;

    // get corresponding id into vector B -- the possibly broadcasted one
    uint idAy = idA / sizeAx;
    uint idAx = idA - (idAy * sizeAx);
    uint idB = getIndex(idAy % sizeBy, idAx % sizeBx, sizeBx);

    // perform computation -- multiply
    vecResult[idA] = vecA[idA] * vecB[idB];
}
)==kernel==";

VectorMultNodeGPUKernel::VectorMultNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memA, cl_mem memB, cl_mem memRes, MemoryDimensions dimA, MemoryDimensions dimB)
    : GPUKernel(compiler.GetComputeUnitCount(), queue, compiler.CompileKernel(KernelSource), inputKernels), _memA(memA), _memB(memB), _memRes(memRes), _dimA(dimA), _dimB(dimB)
{
    assert(((_dimA.xDim == _dimB.xDim) && ((_dimA.yDim % _dimB.yDim == 0) || (_dimB.yDim % _dimA.yDim == 0))) ||
           ((_dimA.yDim == _dimB.yDim) && ((_dimA.xDim % _dimB.xDim == 0) || (_dimB.xDim % _dimA.xDim == 0))));
}

VectorMultNodeGPUKernel::~VectorMultNodeGPUKernel() { }

void VectorMultNodeGPUKernel::Run()
{
    const cl_uint sizeAx = static_cast<cl_uint>(_dimA.xDim);
    const cl_uint sizeBy = static_cast<cl_uint>(_dimB.yDim);
    const cl_uint sizeBx = static_cast<cl_uint>(_dimB.xDim);
    size_t totalWorkItems = _dimA.yDim * _dimA.xDim;
    clSetKernelArg(_kernel, 0, sizeof(cl_mem), &_memA);
    clSetKernelArg(_kernel, 1, sizeof(cl_mem), &_memB);
    clSetKernelArg(_kernel, 2, sizeof(cl_mem), &_memRes);
    clSetKernelArg(_kernel, 3, sizeof(cl_uint), &sizeAx);
    clSetKernelArg(_kernel, 4, sizeof(cl_uint), &sizeBy);
    clSetKernelArg(_kernel, 5, sizeof(cl_uint), &sizeBx);
    clSetKernelArg(_kernel, 6, sizeof(cl_uint), &totalWorkItems);
    std::pair<size_t, size_t> workSize = GetWorkSize(totalWorkItems);
    std::vector<cl_event> inputEvents = GetNodeInputEvents();
    cl_event ownEvent;
    OCLWrappers::CheckCLError(
        clEnqueueNDRangeKernel(_queue, _kernel, 1, nullptr, &(workSize.first), &(workSize.second), inputEvents.size(), inputEvents.data(), &ownEvent)
    , "clEnqueueNDRangeKernel (for VectorMultNodeGPUKernel)");
    SetEvent(ownEvent);
    // todo: can be optimized by having dumb kernel (without any index computations, that get enqueue multiple times with modified arguments?
}
