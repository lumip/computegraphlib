#ifndef _OPENCL_COMPILER_HPP_
#define _OPENCL_COMPILER_HPP_

#include <string>
#include "../OCLWrappers.hpp"

class OpenCLCompiler
{
public:
    OpenCLCompiler() { }
    virtual ~OpenCLCompiler() { }
    virtual OCLWrappers::Kernel CompileKernel(const std::string& kernelSource) = 0;
};

#endif
