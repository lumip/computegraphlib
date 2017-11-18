#include "CompiledGraph.hpp"

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

CompiledGraph::CompiledGraph(std::unique_ptr<GraphCompilationPlatform>&& platform, std::unique_ptr<CompilationMemoryMap>&& CompilationMemoryMap)
    : _platform(std::move(platform))
    , _compilationMemoryMap(std::move(CompilationMemoryMap))
{

}

CompiledGraph::~CompiledGraph() { }

void CompiledGraph::Evaluate(const InputDataMap& inputData)
{
    for (auto input : inputData)
    {
        ConstNodePtr inputNode = _compilationMemoryMap->GetInputNode(input.first);
        SetNodeData(inputNode, input.second);
    }
    _platform->Evaluate();
}

void CompiledGraph::GetOutputData(std::string outputName, DataBuffer& outputBuffer) const
{
    ConstNodePtr outputNode = _compilationMemoryMap->GetOutputNode(outputName);
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
