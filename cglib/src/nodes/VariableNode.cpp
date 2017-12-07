#include "nodes/VariableNode.hpp"

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

VariableNode::VariableNode(std::string name, size_t xDim, size_t yDim)
    : Node(true, true)
    , _name(name)
    , _xDim(xDim)
    , _yDim(yDim)
    , _input(nullptr)
{
}

VariableNode::~VariableNode() {}

void VariableNode::SetInput(const NodePtr inputNode)
{
    if (_input != nullptr)
    {
        this->UnsubscribeFrom(_input);
    }
    _input = inputNode;
    if (_input != nullptr)
    {
        this->SubscribeTo(_input);
    }
}

Node::ConstNodeList VariableNode::GetInputs() const
{
    if (_input != nullptr)
    {
        return ConstNodeList({ _input });
    }
    return ConstNodeList{};
}

std::string VariableNode::ToString() const
{
    return "<VariableNode " + _name + ">";
}

void VariableNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileVariableNode(this);
}

void VariableNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    const MemoryDimensions initDim = memoryMap.GetInputDimensions(_name);
    // todo: what if input node didn't yet register its memory dimensions?
    /*if (_input != nullptr)
    {
        const MemoryDimensions inputNodeDim = memoryMap.GetNodeMemoryDimensions(_input);
        if (initDim != inputNodeDim)
        {
            throw std::runtime_error("The dimensions of initialization data and input node data are different for this VariableNode.");
        }
    }*/
    memoryMap.RegisterNodeMemory(this, initDim);
    memoryMap.RegisterInputMemory(_name, this);
}
