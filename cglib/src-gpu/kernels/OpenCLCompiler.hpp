#ifndef _OPENCL_COMPILER_HPP_
#define _OPENCL_COMPILER_HPP_

#include <string>
#include <vector>
#include <CL/cl.h>

class GPUKernel;

class OpenCLCompiler // rename to OpenCLRuntime ?
{
public:
    OpenCLCompiler() { }
    virtual ~OpenCLCompiler() { }
    virtual cl_kernel CompileKernel(const std::string& kernelSource) = 0;
};

#endif
