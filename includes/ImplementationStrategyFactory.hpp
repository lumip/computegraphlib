#ifndef _IMPLEMENTATION_STRATEGY_FACTORY_HPP_
#define _IMPLEMENTATION_STRATEGY_FACTORY_HPP_

#include <memory>
class GraphCompilationPlatform;
class NodeCompiler;
class CompilationMemoryMap;

class ImplementationStrategyFactory
{
public:
    std::unique_ptr<GraphCompilationPlatform> CreateGraphCompilationTargetStrategy(CompilationMemoryMap& CompilationMemoryMap) const;
    //std::unique_ptr<NodeCompiler> CreateNodeCompiler(GraphCompilationPlatform* strategy) const;
};

#endif
