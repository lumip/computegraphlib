#ifdef GPU
#include "nodes/InputNode.hpp"

void InputNode::Compile(GraphCompilationContext& context) const
{
    throw std::logic_error("not implemented yet.");
}
#endif
