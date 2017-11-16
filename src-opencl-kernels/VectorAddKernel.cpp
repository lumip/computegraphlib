#include <string>

std::string VectorAddKernelSource = R"==kernel==(
__kernel void main(__global float* vecA, __global float* vecB, __global float* vecResult)
{
    int gid = get_global_id(0);
    vecResult[gid] = vecA[gid] + vecB[gid];
}
)==kernel==";

std::string MatrixMultSource = R"==kernel==(
__kernel void main(__global float* matA, __global float* matB, __global float* matResult, uint dim)
{
    uint i = get_global_id(0);
    uint j = get_global_id(1);
    uint m = get_global_size(0);
    uint n = get_global_size(1);
    float val = 0.0f;
    for (uint k = 0; k < dim; ++k)
    {
        val += matA[i * dim + k] * matB[k * n + j];
    }
    matResult[i * n + j] = val;
}
)==kernel==";
