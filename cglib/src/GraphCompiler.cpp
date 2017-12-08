#include "GraphCompiler.hpp"

#include <assert.h>

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

    const std::vector<ConstNodePtr> nodeTopology = DetermineNodeOrder(outputNode);
    for (ConstNodePtr node : nodeTopology)
    {
        node->GetMemoryDimensions(*context);
    }

    for (ConstNodePtr node : nodeTopology)
    {
        // if node can operate in-place, try to find an input node that is used only by this node and reuse it's memory buffer (if size is sufficient)
        if (node->CanOperateInPlace())
        {
            const MemoryDimensions nodeDim = context->GetNodeMemoryDimensions(node);
            for (ConstNodePtr inputNode : node->GetInputs())
            {
                if (inputNode->GetSubscribers().size() == 1)
                {
                    const MemoryDimensions inputDim = context->GetNodeMemoryDimensions(inputNode);
                    if (nodeDim <= inputDim)
                    {
                        // if our desired input node does not have a memory assignment yet, we allocate some for it! *being generous*
                        if (!strategy->NodeIsAssigned(inputNode))
                        {
                            strategy->ReserveMemoryBuffer(inputNode);
                        }
                        const GraphCompilationPlatform::MemoryBufferHandle handle = strategy->GetNodeMemoryBuffer(inputNode);
                        strategy->AssignMemoryBuffer(node, handle); // note that we could already have some memory assigned to us already here, but that is not a problem
                                                                   // (old and new memory locatiosn will be merged by AssignNodeMemory() call)
                        break; // after finding a suitable input buffer for reuse, leave the loop (important! cannot reassign anymore)
                    }
                }
            }
        }
        // if node has no memory assigned from previous check, allocate new memory block
        if (!strategy->NodeIsAssigned(node))
        {
            strategy->ReserveMemoryBuffer(node);
        }
    }

    strategy->AllocateAllMemory();


    // todo: figure out how to check output and input size of variable node for consistency.
    for (ConstNodePtr node : nodeTopology)
    {
        assert(strategy->NodeIsAssigned(node));
        node->Compile(*strategy);
    }
    return std::unique_ptr<CompiledGraph>(new CompiledGraph(std::move(strategy), std::move(context)));
}
