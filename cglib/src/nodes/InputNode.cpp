#include "nodes/InputNode.hpp"

#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"

InputNode::InputNode(std::string name)
    : Node(true, true)
    , _name(name)
{
}

InputNode::~InputNode() { }

Node::ConstNodeList InputNode::GetInputs() const
{
    return ConstNodeList();
}

const std::string& InputNode::GetName() const
{
    return _name;
}

std::string InputNode::ToString() const
{
    return "<InputNode " + _name + ">";
}

void InputNode::Compile(GraphCompilationPlatform& platform) const
{
    platform.CompileInputNode(this);
}

void InputNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    const MemoryDimensions dim = memoryMap.GetInputDimensions(_name);
    memoryMap.RegisterNodeMemory(this, dim);
    memoryMap.RegisterInputNode(_name, this);
}
