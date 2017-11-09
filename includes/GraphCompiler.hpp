#ifndef _GRAPH_COMPILER_HPP_
#define _GRAPH_COMPILER_CPP_

#include <vector>
#include <memory>

#include "nodes/Node.hpp"
#include "GraphCompilationContext.hpp"
#include "CompiledGraph.hpp"

class GraphCompiler
{
private:
    GraphCompilationContext _context;
private:
    void VisitNode(const ConstNodePtr node, std::vector<ConstNodePtr>& nodeTopology, std::set<ConstNodePtr>& visitedNodes) const;
    std::vector<ConstNodePtr> DetermineNodeOrder(const ConstNodePtr outputNode) const;
public:
    GraphCompiler();
    virtual ~GraphCompiler();
    std::unique_ptr<const CompiledGraph> Compile(const ConstNodePtr outputNode, const InputDataMap& inputData);
};

#endif
