#ifdef GPU
#include "nodes/VectorAddNode.hpp"

std::unique_ptr<const Kernel> VectorAddNode::Compile(GraphCompilationContext* const context) const
{
    throw std::logic_error("not implemented yet.");
}
#endif
