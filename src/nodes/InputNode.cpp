#include "nodes/InputNode.hpp"
#include "CompilationMemoryMap.hpp"

InputNode::InputNode(std::string name, size_t dim)
    : _name(name)
    , _dim(dim)
{
}

InputNode::~InputNode() { }

Node::ConstNodeList InputNode::GetInputs() const
{
    return ConstNodeList();
}

std::string InputNode::ToString() const
{
    return "<InputNode " + _name + ">";
}

bool InputNode::IsInitialized() const
{
    return true;
}

void InputNode::Compile(GraphCompilationPlatform& platform) const
{
}

void InputNode::GetMemoryDimensions(CompilationMemoryMap& memoryMap) const
{
    const MemoryDimensions dim = memoryMap.GetInputDimensions(_name);
    if (dim.xDim != this->_dim)
    {
        throw std::invalid_argument("Dimension sizes for input " + _name + " differ between node declaration and input map delcaration.");
    }
    memoryMap.RegisterNodeMemory(this, dim);
    memoryMap.RegisterInputMemory(_name, this);
}
