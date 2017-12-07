#ifndef _GRAPH_COMPILER_HPP_
#define _GRAPH_COMPILER_CPP_

#include <vector>
#include <memory>
#include <set>
#include <queue>

#include <types.hpp>

class ImplementationStrategyFactory;
class CompiledGraph;
class CompilationMemoryMap;

class GraphCompiler
{
private:
    const std::unique_ptr<const ImplementationStrategyFactory> _strategyFactory;
private:
    void VisitNode(const ConstNodePtr node, std::vector<ConstNodePtr>& nodeTopology, std::set<ConstNodePtr>& visitedNodes, std::queue<ConstNodePtr>& seedNodes) const;
    std::vector<ConstNodePtr> DetermineNodeOrder(const ConstNodePtr outputNode) const;
public:
    GraphCompiler(std::unique_ptr<const ImplementationStrategyFactory>&& strategyFactory);
    virtual ~GraphCompiler();
    std::unique_ptr<CompiledGraph> Compile(const ConstNodePtr outputNode, const InputDimensionsMap& inputDimensions) const;
};

#endif
