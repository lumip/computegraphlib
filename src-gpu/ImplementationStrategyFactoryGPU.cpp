#ifdef GPU

#include "ImplementationStrategyFactory.hpp"
#include "GraphCompilationGPUStrategy.hpp"

std::unique_ptr<GraphCompilationTargetStrategy> ImplementationStrategyFactory::CreateGraphCompilationTargetStrategy() const
{
    return std::unique_ptr<GraphCompilationTargetStrategy>(new GraphCompilationGPUStrategy);
}

#endif
