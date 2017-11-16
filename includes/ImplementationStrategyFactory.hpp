#ifndef _IMPLEMENTATION_STRATEGY_FACTORY_HPP_
#define _IMPLEMENTATION_STRATEGY_FACTORY_HPP_

#include <memory>
class GraphCompilationTargetStrategy;
class NodeCompiler;

class ImplementationStrategyFactory
{
public:
    std::unique_ptr<GraphCompilationTargetStrategy> CreateGraphCompilationTargetStrategy() const;
    std::unique_ptr<NodeCompiler> CreateNodeCompiler(GraphCompilationTargetStrategy* strategy) const;
};

#endif
