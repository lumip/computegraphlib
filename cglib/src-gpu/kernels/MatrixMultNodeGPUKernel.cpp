#include "MatrixMultNodeGPUKernel.hpp"

#include <assert.h>

#include "OpenCLCompiler.hpp"
#include "../OCLWrappers.hpp"

// performance with 0sc03g (1000x10000 and 4000x10000) and no local workgroup size set
// b global  / a global : 4.012 s / 7.804 s
// b private / a global : 3.658 s / 7.831 s
// b global  / a local  : 2.918 s / 7.795 s
// b private / a local  : 2.662 s / 7.692 s
const std::string MatrixMultNodeGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float* matA, __global float* matB, __global float* matResult, uint m, uint n, uint d, __local float* a_i)
{
    uint j = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsiz = get_local_size(0);

    for (uint i = 0; i < m; ++i)
    {
        for (uint k = lid; k < d; k += lsiz)
        {
            a_i[k] = matA[i * d + k];
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        if (j < n)
        {
            float val = 0.0f;
            for (uint k = 0; k < d; ++k)
            {
                val += a_i[k] * matB[k * n + j];
            }
            matResult[i * n + j] = val;
        }
    }
}
)==kernel==";

//// computing rows per work items is less efficient than computing columns per work item (probable cause: memory access to not coalesce as nicely)
//// (the effect is pretty negligible with local+private memory, though)
//// performance with 0sc03g (1000x10000) and no local workgroup size set
//// a global  / b global : 8.318 s
//// a private / b global : 4.862 s
//// a global  / b local  : 8.137 s
//// a private / b local  : 2.764 s
//const std::string MatrixMultNodeGPUKernel::KernelSource = R"==kernel==(
//__kernel void main(__global float* matA, __global float* matB, __global float* matResult, uint m, uint n, uint d, __local float* b_j)
//{
//    uint i = get_global_id(0);
//    uint lid = get_local_id(0);
//    uint lsiz = get_local_size(0);

//    //float a_i[10000];
//    //for (uint k = 0; k < d; ++k)
//    //{
//    //    a_i[k] = matA[i * d + k];
//    //}

//    for (uint j = 0; j < n; ++j)
//    {
//        //for (uint k = lid; k < d; k += lsiz)
//        //{
//        //    b_j[k] = matB[k * n + j];
//        //}
//        //work_group_barrier(CLK_LOCAL_MEM_FENCE);

//        float val = 0.0f;
//        for (uint k = 0; k < d; ++k)
//        {
//            //val += a_i[k] * b_j[k];
//            //val += a_i[k] * matB[k * n + j];
//            //val += matA[i * d + k] * b_j[k];
//            val += matA[i * d + k] * matB[k * n + j];
//        }
//        matResult[i * n + j] = val;
//    }
//}
//)==kernel==";

/*
const std::string MatrixMultNodeGPUKernel::KernelSource = R"==kernel==(
__kernel void main(__global float* matA, __global float* matB, __global float* matResult, uint n, uint d)
{
    uint i = get_global_id(0);

    for (uint j = 0; j < n; ++j)
    {
        float val = 0.0f;
        for (uint k = 0; k < d; ++k)
        {
            val += matA[i * d + k] * matB[k * n + j];
        }
        matResult[i * n + j] = val;
    }
}
)==kernel==";*/

MatrixMultNodeGPUKernel::MatrixMultNodeGPUKernel(OpenCLCompiler& compiler, const cl_command_queue queue, const GPUKernel::ConstList& inputKernels, cl_mem memA, cl_mem memB, cl_mem memRes, size_t m, size_t n, size_t d)
    : GPUKernel(queue, compiler.CompileKernel(KernelSource), inputKernels), _memA(memA), _memB(memB), _memRes(memRes), _m(m), _n(n), _d(d)
{ }

MatrixMultNodeGPUKernel::~MatrixMultNodeGPUKernel() { }

void MatrixMultNodeGPUKernel::Run()
{
    clSetKernelArg(_kernel, 0, sizeof(cl_mem), &_memA);
    clSetKernelArg(_kernel, 1, sizeof(cl_mem), &_memB);
    clSetKernelArg(_kernel, 2, sizeof(cl_mem), &_memRes);
    clSetKernelArg(_kernel, 3, sizeof(cl_uint), &_m);
    clSetKernelArg(_kernel, 4, sizeof(cl_uint), &_n);
    clSetKernelArg(_kernel, 5, sizeof(cl_uint), &_d);
    clSetKernelArg(_kernel, 6, sizeof(float) * _d, nullptr);
    std::pair<size_t, size_t> workSize = GetWorkSize(_n);
    std::vector<cl_event> inputEvents = GetNodeInputEvents();
    cl_event ownEvent;
    OCLWrappers::CheckCLError(
        clEnqueueNDRangeKernel(_queue, _kernel, 1, nullptr, &(workSize.first), &(workSize.second), inputEvents.size(), inputEvents.data(), &ownEvent)
    , "clEnqueueNDRangeKernel (for MatrixMultNodeGPUKernel)");
    SetEvent(ownEvent);
}
