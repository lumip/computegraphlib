#ifndef _IMPLEMENTATION_STRATEGY_FACTORY_HPP_
#define _IMPLEMENTATION_STRATEGY_FACTORY_HPP_

#include <memory>

class GraphCompilationPlatform;
class CompilationMemoryMap;

class ImplementationStrategyFactory
{
public:
    std::unique_ptr<GraphCompilationPlatform> CreateGraphCompilationTargetStrategy(CompilationMemoryMap& CompilationMemoryMap) const;
};

#endif
