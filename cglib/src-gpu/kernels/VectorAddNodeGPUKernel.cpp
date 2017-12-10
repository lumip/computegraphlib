#include "VectorAddNodeGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"
#include "../OCLWrappers.hpp"

const std::string VectorAddNodeGPUKernel::KernelSource = R"==kernel==(
uint getIndex(uint y, uint x, uint stride)
{
    return y * stride + x;
}

__kernel void main(__global float* vecA, __global float* vecB, __global float* vecResult, uint sizeAx, uint sizeBy, uint sizeBx)
{
    // get id into vector A -- the bigger one
    uint idAy = get_global_id(0);
    uint idAx = get_global_id(1);
    // get corresponding id into vector B -- to possibly broadcasted one
    uint idA = getIndex(idAy, idAx, sizeAx);
    uint idB = getIndex(idAy % sizeBy, idAx % sizeBx, sizeBx);

    // perform computation -- add
    vecResult[idA] = vecA[idA] + vecB[idB];
}
)==kernel==";

VectorAddNodeGPUKernel::VectorAddNodeGPUKernel(OpenCLCompiler& compiler, cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memA, cl_mem memB, cl_mem memRes, MemoryDimensions dimA, MemoryDimensions dimB)
    : GPUKernel(queue, compiler.CompileKernel(KernelSource), inputKernels), _memA(memA), _memB(memB), _memRes(memRes), _dimA(dimA), _dimB(dimB)
{
    assert(((_dimA.xDim == _dimB.xDim) && ((_dimA.yDim % _dimB.yDim == 0) || (_dimB.yDim % _dimA.yDim == 0))) ||
           ((_dimA.yDim == _dimB.yDim) && ((_dimA.xDim % _dimB.xDim == 0) || (_dimB.xDim % _dimA.xDim == 0))));
}

VectorAddNodeGPUKernel::~VectorAddNodeGPUKernel() { }

void VectorAddNodeGPUKernel::Run()
{
    const cl_uint sizeAx = static_cast<cl_uint>(_dimA.xDim);
    const cl_uint sizeBy = static_cast<cl_uint>(_dimB.yDim);
    const cl_uint sizeBx = static_cast<cl_uint>(_dimB.xDim);
    clSetKernelArg(_kernel, 0, sizeof(cl_mem), &_memA);
    clSetKernelArg(_kernel, 1, sizeof(cl_mem), &_memB);
    clSetKernelArg(_kernel, 2, sizeof(cl_mem), &_memRes);
    clSetKernelArg(_kernel, 3, sizeof(cl_uint), &sizeAx);
    clSetKernelArg(_kernel, 4, sizeof(cl_uint), &sizeBy);
    clSetKernelArg(_kernel, 5, sizeof(cl_uint), &sizeBx);
    size_t globalWorkSize[2] = { _dimA.yDim, _dimA.xDim };
    std::vector<cl_event> inputEvents = GetNodeInputEvents();
    cl_event ownEvent;
    OCLWrappers::CheckCLError(
        clEnqueueNDRangeKernel(_queue, _kernel, 2, nullptr, globalWorkSize, nullptr, inputEvents.size(), inputEvents.data(), &ownEvent)
    , "clEnqueueNDRangeKernel (for VectorAddNodeGPUKernel)");
    SetEvent(ownEvent);
    // todo: can be optimized by having dumb kernel (without any index computations, that get enqueue multiple times with modified arguments?
}
