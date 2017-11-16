#include "ImplementationStrategyFactory.hpp"
#include "GraphCompilationCPUStrategy.hpp"
#include "NodeCompilerCPU.hpp"

std::unique_ptr<GraphCompilationPlatform> ImplementationStrategyFactory::CreateGraphCompilationTargetStrategy() const
{
    return std::unique_ptr<GraphCompilationPlatform>(new GraphCompilationCPUStrategy);
}

std::unique_ptr<NodeCompiler> ImplementationStrategyFactory::CreateNodeCompiler(GraphCompilationPlatform* strategy) const
{
    return std::unique_ptr<NodeCompiler>(new NodeCompilerCPU());
}
