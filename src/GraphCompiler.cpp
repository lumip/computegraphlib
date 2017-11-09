#include "GraphCompiler.hpp"
#include "nodes/Node.hpp"
#include "nodes/VariableNode.hpp"

GraphCompiler::GraphCompiler()
    : _context(InputDataMap())
{

}

GraphCompiler::~GraphCompiler() { }

void GraphCompiler::VisitNode(const ConstNodePtr node, std::vector<ConstNodePtr>& nodeTopology, std::set<ConstNodePtr>& visitedNodes) const
{
    if (visitedNodes.find(node) == visitedNodes.end())
    {
        visitedNodes.insert(node);
        if (!node->IsInitialized())
        {
            for (ConstNodePtr childNode : node->GetInputs())
            {
                VisitNode(childNode, nodeTopology, visitedNodes);
            }
        }
        nodeTopology.push_back(node);
    }
}

std::vector<ConstNodePtr> GraphCompiler::DetermineNodeOrder(const ConstNodePtr outputNode) const
{
    std::vector<ConstNodePtr> nodeTopology;
    std::set<ConstNodePtr> visitedNodes;
    VisitNode(outputNode, nodeTopology, visitedNodes);
    return nodeTopology;
}

std::unique_ptr<const CompiledGraph> GraphCompiler::Compile(const ConstNodePtr outputNode, const InputDataMap& inputData)
{
    std::unique_ptr<GraphCompilationContext> context(new GraphCompilationContext(inputData));

    const std::vector<ConstNodePtr> nodeTopology = DetermineNodeOrder(outputNode);
    for (size_t i = 0; i < nodeTopology.size(); ++i)
    {
        nodeTopology[i]->Compile(*context);
    }
    return std::move(context);
}
