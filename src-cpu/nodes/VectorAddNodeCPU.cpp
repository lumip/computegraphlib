#ifdef CPU
#include <iostream>
#include "nodes/VectorAddNode.hpp"
#include "GraphCompilationContext.hpp"

void VectorAddNode::Compile(GraphCompilationContext* const context) const
{
    std::cout << "VectorAddNode compiling." << std::endl;
}
#endif
