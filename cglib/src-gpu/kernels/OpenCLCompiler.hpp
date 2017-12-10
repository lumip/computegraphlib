#ifndef _OPENCL_COMPILER_HPP_
#define _OPENCL_COMPILER_HPP_

#include <string>

class OpenCLCompiler
{
public:
    OpenCLCompiler() { }
    virtual ~OpenCLCompiler() { }
    virtual cl_kernel CompileKernel(const std::string& kernelSource) = 0;
};

#endif
