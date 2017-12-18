#include "GraphCompilationPlatformFactory.hpp"

#include "GraphCompilationCPUPlatform.hpp"

std::unique_ptr<GraphCompilationPlatform> GraphCompilationPlatformFactory::CreateGraphCompilationTargetStrategy(CompilationMemoryMap& CompilationMemoryMap) const
{
    return std::unique_ptr<GraphCompilationPlatform>(new GraphCompilationCPUPlatform(CompilationMemoryMap));
}
