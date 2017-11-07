#ifndef _GRAPH_COMPILER_HPP_
#define _GRAPH_COMPILER_CPP_

#include <vector>

#include "nodes/Node.hpp"
#include "GraphCompilationContext.hpp"

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
    std::vector<std::unique_ptr<const Kernel>> Compile(const ConstNodePtr outputNode, const InputDataMap& inputData); // todo: change this. for OpenCL we do not want many separate kernels that only then set up the GPU.. all that should happend within Compile().
};

#endif
