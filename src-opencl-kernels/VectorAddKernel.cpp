#include <string>

namespace kernel_sources
{

std::string VectorAddKernel = R"==kernel==("
__kernel VectorAdd(__global float* vecA, __global float* vecB, __global float* vecResult)
{
  int gid = get_global_id(0);
  vecResult[gid] = vecA[gid] + vecB[gid];
}
)==kernel==";

}
