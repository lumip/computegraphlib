#ifdef CPU
#include <iostream>
#include "nodes/InputNode.hpp"
#include "GraphCompilationContext.hpp"
#include "Kernel.hpp"

class InputNodeCPUKernel : public Kernel
{
public:
    InputNodeCPUKernel() { }
    ~InputNodeCPUKernel() { }
    void Run() { }
};

std::unique_ptr<const Kernel> InputNode::Compile(GraphCompilationContext* const context) const
{
    std::cout << "InputNode " << this->_name << " compiling." << std::endl;
    InputDataBuffer input = context->GetInputDataBuffer(_name);
    return std::unique_ptr<const Kernel>(new InputNodeCPUKernel());  // std::make_unique only since c++14
}
#endif
