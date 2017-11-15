#include <CL/cl.h>

#include "ImplementationStrategyFactory.hpp"
#include "GraphCompilationGPUStrategy.hpp"

std::unique_ptr<GraphCompilationTargetStrategy> ImplementationStrategyFactory::CreateGraphCompilationTargetStrategy() const
{
    cl_platform_id platformId;
    GraphCompilationGPUStrategy::CheckCLError(clGetPlatformIDs(1, &platformId, nullptr), "clGetPlatformIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformId, 0
    };
    cl_int status = CL_SUCCESS;
    cl_context context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
    GraphCompilationGPUStrategy::CheckCLError(status, "clCreateContextFromType");

    return std::unique_ptr<GraphCompilationTargetStrategy>(new GraphCompilationGPUStrategy(context));
}
