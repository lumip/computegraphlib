#ifdef GPU
#include "nodes/InputNode.hpp"

std::unique_ptr<const Kernel> InputNode::Compile(GraphCompilationContext* const context) const
{
    throw std::logic_error("not implemented yet.");
}
#endif
