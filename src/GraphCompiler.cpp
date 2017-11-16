#include "GraphCompiler.hpp"
#include "nodes/Node.hpp"
#include "NodeCompiler.hpp"

GraphCompiler::GraphCompiler(std::unique_ptr<const ImplementationStrategyFactory>&& strategyFactory)
    : _strategyFactory(std::move(strategyFactory))
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

std::unique_ptr<CompiledGraph> GraphCompiler::Compile(const ConstNodePtr outputNode, const InputDimensionsMap& inputDimensions) const
{
    std::unique_ptr<GraphCompilationTargetStrategy> strategy = _strategyFactory->CreateGraphCompilationTargetStrategy();
    std::unique_ptr<NodeCompiler> nodeCompiler(_strategyFactory->CreateNodeCompiler(&(*strategy)));
    std::unique_ptr<GraphCompilationContext> context(new GraphCompilationContext(inputDimensions, std::move(strategy)));

    const std::vector<ConstNodePtr> nodeTopology = DetermineNodeOrder(outputNode);
    // todo: add functionality which determines which nodes need separate memory and which do not (temporary results only used in one other node can be overwritten)
    for (size_t i = 0; i < nodeTopology.size(); ++i)
    {
        nodeTopology[i]->Compile(*context, *nodeCompiler);
    }
    return std::move(context);
}
