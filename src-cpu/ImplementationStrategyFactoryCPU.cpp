#ifdef CPU

#include "ImplementationStrategyFactory.hpp"
#include "GraphCompilationCPUStrategy.hpp"

std::unique_ptr<GraphCompilationTargetStrategy> ImplementationStrategyFactory::CreateGraphCompilationTargetStrategy() const
{
    return std::unique_ptr<GraphCompilationTargetStrategy>(new GraphCompilationCPUStrategy);
}

#endif
