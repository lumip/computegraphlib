#include "nodes/InputNode.hpp"
#include "GraphCompilationContext.hpp"
#include "NodeCompiler.hpp"

void InputNode::Compile(GraphCompilationContext& context, NodeCompiler& nodeCompiler) const
{
    const MemoryDimensions dim = context.GetInputDimensions(_name);
    if (dim.xDim != this->_dim)
    {
        throw std::invalid_argument("Dimension size for input " + _name + "differ between node declaration and input map declaration.");
    }
    const NodeMemoryDescriptor memDesc = context.RegisterMemory(dim);
    context.AssignNodeMemory(this, memDesc.handle);
    context.RegisterInputMemory(_name, memDesc.handle);
}
