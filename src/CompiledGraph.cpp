#include "CompiledGraph.hpp"
#include "GraphCompilationPlatform.hpp"
#include "GraphCompilationContext.hpp"

CompiledGraph::CompiledGraph(std::unique_ptr<GraphCompilationPlatform>&& platform, std::unique_ptr<MemoryCompilationMap>&& memoryCompilationMap)
    : _platform(std::move(platform))
    , _memoryCompilationMap(std::move(memoryCompilationMap))
{

}

CompiledGraph::~CompiledGraph() { }

void CompiledGraph::Evaluate(const InputDataMap& inputData)
{
    for (auto input : inputData)
    {
        ConstNodePtr inputNode = _memoryCompilationMap->GetInputNode(input.first);
        SetNodeData(inputNode, input.second);
    }
    _platform->Evaluate();
}

void CompiledGraph::GetOutputData(std::string outputName, DataBuffer& outputBuffer) const
{
    ConstNodePtr outputNode = _memoryCompilationMap->GetOutputNode(outputName);
    GetNodeData(outputNode, outputBuffer);
}

void CompiledGraph::GetNodeData(const ConstNodePtr node, DataBuffer& outputBuffer) const
{
    _platform->CopyOutputData(node, outputBuffer);
}

void CompiledGraph::SetNodeData(const ConstNodePtr node, InputDataBuffer& inputBuffer)
{
    _platform->CopyInputData(node, inputBuffer);
}
