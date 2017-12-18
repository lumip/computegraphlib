#include "GraphCompilationGPUPlatform.hpp"

#include <CL/cl.h>

#include "GraphCompilationPlatformFactory.hpp"

#include "OCLWrappers.hpp"

std::unique_ptr<GraphCompilationPlatform> GraphCompilationPlatformFactory::CreateGraphCompilationTargetStrategy(CompilationMemoryMap& CompilationMemoryMap) const
{
    cl_platform_id platformId;
    OCLWrappers::CheckCLError(clGetPlatformIDs(1, &platformId, nullptr), "clGetPlatformIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformId, 0
    };
    cl_int status = CL_SUCCESS;
    cl_context raw_context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &status);

    OCLWrappers::CheckCLError(status, "clCreateContextFromType");

    return std::unique_ptr<GraphCompilationPlatform>(new GraphCompilationGPUPlatform(CompilationMemoryMap, OCLWrappers::Context(raw_context)));
}
