#include "CompiledGraph.hpp"

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"
#include "nodes/Node.hpp"

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

bool CompiledGraph::IsEvaluating() const
{
    return _platform->IsEvaluating();
}

void CompiledGraph::WaitUntilFinished() const
{
    _platform->WaitUntilEvaluationFinished();
}

void CompiledGraph::InitializeVariables(const InputDataMap& inputData)
{
    for (auto input : inputData)
    {
        ConstNodePtr variableNode = _compilationMemoryMap->GetVariableNode(input.first);
        Node::ConstNodeList nodeInputs = variableNode->GetInputs();
        if (nodeInputs.size() > 0)
        {
            // if VariableNode has input, it either shares memory with that or copies its input memories when processed. either way, we initialize the data into the input's buffer
            SetNodeData(nodeInputs[0], input.second);
        }
        else
        {
            // if VariableNode has no input, we initialize the data into it's own buffer
            SetNodeData(variableNode, input.second);
        }
    }
}

void CompiledGraph::GetNodeData(const ConstNodePtr node, float* outputBuffer) const
{
    _platform->CopyOutputData(node, outputBuffer);
}

void CompiledGraph::SetNodeData(const ConstNodePtr node, float const* inputBuffer)
{
    _platform->CopyInputData(node, inputBuffer);
}

float* CompiledGraph::GetMappedInputBuffer(std::string const& inputName)
{
    return _platform->GetMappedInputBuffer(inputName);
}
