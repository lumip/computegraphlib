#ifdef CPU
#include <memory>

#include "nodes/VariableNode.hpp"
#include "Kernel.hpp"

class VariableNodeCPUKernel : public Kernel
{
public:
    VariableNodeCPUKernel() { }
    ~VariableNodeCPUKernel() { }
    void Run() const { }
};

std::unique_ptr<const Kernel> VariableNode::Compile(GraphCompilationContext* const context) const
{
    return std::unique_ptr<const Kernel>(new VariableNodeCPUKernel());  // std::make_unique only since c++14
}

#endif
