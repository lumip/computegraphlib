#include "ImplementationStrategyFactory.hpp"
#include "GraphCompilationCPUStrategy.hpp"
#include "NodeCompilerCPU.hpp"

std::unique_ptr<GraphCompilationPlatform> ImplementationStrategyFactory::CreateGraphCompilationTargetStrategy(MemoryCompilationMap& memoryCompilationMap) const
{
    return std::unique_ptr<GraphCompilationPlatform>(new GraphCompilationCPUStrategy(memoryCompilationMap));
}
