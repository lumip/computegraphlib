#ifdef GPU
#include "nodes/VariableNode.hpp"

std::unique_ptr<const Kernel> VariableNode::Compile(GraphCompilationContext* const context) const
{
    throw std::logic_error("not implemented yet.");
}

#endif
