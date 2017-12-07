#include "GraphCompiler.hpp"

#include <queue>

#include "nodes/Node.hpp"
#include "CompilationMemoryMap.hpp"
#include "CompiledGraph.hpp"
#include "GraphCompilationPlatform.hpp"
#include "ImplementationStrategyFactory.hpp"

GraphCompiler::GraphCompiler(std::unique_ptr<const ImplementationStrategyFactory>&& strategyFactory)
    : _strategyFactory(std::move(strategyFactory))
{

}

GraphCompiler::~GraphCompiler() { }

void GraphCompiler::VisitNode(const ConstNodePtr node, std::vector<ConstNodePtr>& nodeTopology, std::set<ConstNodePtr>& visitedNodes, std::queue<ConstNodePtr>& seedNodes) const
{
    if (visitedNodes.find(node) == visitedNodes.end())
    {
        visitedNodes.insert(node);
        if (!node->IsInitialized())
        {
            for (ConstNodePtr childNode : node->GetInputs())
            {
                VisitNode(childNode, nodeTopology, visitedNodes, seedNodes);
            }
        }
        else
        {
            for (ConstNodePtr childNode : node ->GetInputs())
            {
                seedNodes.push(childNode);
            }
        }
        nodeTopology.push_back(node);
    }
}

std::vector<ConstNodePtr> GraphCompiler::DetermineNodeOrder(const ConstNodePtr outputNode) const
{
    std::queue<ConstNodePtr> seedNodes;
    seedNodes.push(outputNode);
    std::vector<ConstNodePtr> nodeTopology;
    std::set<ConstNodePtr> visitedNodes;
    while (!seedNodes.empty())
    {
        ConstNodePtr node = seedNodes.front();
        seedNodes.pop();
        VisitNode(node, nodeTopology, visitedNodes, seedNodes);
    }
    return nodeTopology;
}

std::unique_ptr<CompiledGraph> GraphCompiler::Compile(const ConstNodePtr outputNode, const InputDimensionsMap& inputDimensions) const
{
    std::unique_ptr<CompilationMemoryMap> context(new CompilationMemoryMap(inputDimensions));
    std::unique_ptr<GraphCompilationPlatform> strategy = _strategyFactory->CreateGraphCompilationTargetStrategy(*context);
    //std::unique_ptr<NodeCompiler> nodeCompiler(_strategyFactory->CreateNodeCompiler(&(*strategy)));

    const std::vector<ConstNodePtr> nodeTopology = DetermineNodeOrder(outputNode);
    // todo: add functionality which determines which nodes need separate memory and which do not (temporary results only used in one other node can be overwritten)
    for (size_t i = 0; i < nodeTopology.size(); ++i)
    {
        ConstNodePtr node = nodeTopology[i];
        node->GetMemoryDimensions(*context);
        strategy->AllocateMemory(node);
    }
    // todo: figure out how to check output and input size of variable node for consistency.
    for (size_t i = 0; i < nodeTopology.size(); ++i)
    {
        nodeTopology[i]->Compile(*strategy);
    }
    return std::unique_ptr<CompiledGraph>(new CompiledGraph(std::move(strategy), std::move(context)));
}
