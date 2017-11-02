#ifdef CPU
#include <iostream>
#include "nodes/InputNode.hpp"
#include "GraphCompilationContext.hpp"

void InputNode::Compile(GraphCompilationContext* const context) const
{
    std::cout << "InputNode " << this->_name << " compiling." << std::endl;
}
#endif
