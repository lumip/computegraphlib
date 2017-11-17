#include <CL/cl.h>

#include "ImplementationStrategyFactory.hpp"
#include "GraphCompilationGPUPlatform.hpp"

std::unique_ptr<GraphCompilationPlatform> ImplementationStrategyFactory::CreateGraphCompilationTargetStrategy(CompilationMemoryMap& CompilationMemoryMap) const
{
    cl_platform_id platformId;
    GraphCompilationGPUPlatform::CheckCLError(clGetPlatformIDs(1, &platformId, nullptr), "clGetPlatformIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformId, 0
    };
    cl_int status = CL_SUCCESS;
    cl_context context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &status);
    GraphCompilationGPUPlatform::CheckCLError(status, "clCreateContextFromType");

    return std::unique_ptr<GraphCompilationPlatform>(new GraphCompilationGPUPlatform(CompilationMemoryMap, context));
}
