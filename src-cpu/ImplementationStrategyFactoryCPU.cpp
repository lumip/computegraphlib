#include "ImplementationStrategyFactory.hpp"
#include "GraphCompilationCPUPlatform.hpp"

std::unique_ptr<GraphCompilationPlatform> ImplementationStrategyFactory::CreateGraphCompilationTargetStrategy(CompilationMemoryMap& CompilationMemoryMap) const
{
    return std::unique_ptr<GraphCompilationPlatform>(new GraphCompilationCPUPlatform(CompilationMemoryMap));
}
