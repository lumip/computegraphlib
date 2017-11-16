#ifndef _IMPLEMENTATION_STRATEGY_FACTORY_HPP_
#define _IMPLEMENTATION_STRATEGY_FACTORY_HPP_

#include <memory>
class GraphCompilationPlatform;
class NodeCompiler;

class ImplementationStrategyFactory
{
public:
    std::unique_ptr<GraphCompilationPlatform> CreateGraphCompilationTargetStrategy() const;
    std::unique_ptr<NodeCompiler> CreateNodeCompiler(GraphCompilationPlatform* strategy) const;
};

#endif
