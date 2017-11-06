#ifdef GPU
#include "nodes/MatrixMultNode.hpp"

std::unique_ptr<const Kernel> MatrixMultNode::Compile(GraphCompilationContext* const context) const
{
    throw std::logic_error("not implemented yet.");
}
#endif
