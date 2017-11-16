#include "nodes/InputNode.hpp"

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

void InputNode::Compile(GraphCompilationContext &context, NodeCompiler &nodeCompiler) const
{
    const MemoryDimensions dim = context.GetInputDimensions(_name);
    if (dim.xDim != this->_dim)
    {
        throw std::invalid_argument("Dimension sizes for input " + _name + " differ between node declaration and input map declaration.");
    }
    const NodeMemoryDescriptor memDesc = context.AllocateMemory(dim);
    context.AssignNodeMemory(this, memDesc.handle);
    context.RegisterInputMemory(_name, memDesc.handle);
}

MemoryDimensions InputNode::GetMemoryDimensions(const InputDimensionsMap& inputDimensions, const std::map<ConstNodePtr, MemoryDimensions>& nodeMemoryDimensions) const
{
    const MemoryDimensions dim = inputDimensions.at(_name);
    if (dim.xDim != this->_dim)
    {
        throw std::invalid_argument("Dimension sizes for input " + _name + " differ between node declaration and input map delcaration.");
    }
    return dim;
}
