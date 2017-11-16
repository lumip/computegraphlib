#include "ImplementationStrategyFactory.hpp"
#include "GraphCompilationCPUStrategy.hpp"
#include "NodeCompilerCPU.hpp"

std::unique_ptr<GraphCompilationTargetStrategy> ImplementationStrategyFactory::CreateGraphCompilationTargetStrategy() const
{
    return std::unique_ptr<GraphCompilationTargetStrategy>(new GraphCompilationCPUStrategy);
}

std::unique_ptr<NodeCompiler> ImplementationStrategyFactory::CreateNodeCompiler(GraphCompilationTargetStrategy* strategy) const
{
    return std::unique_ptr<NodeCompiler>(new NodeCompilerCPU());
}
