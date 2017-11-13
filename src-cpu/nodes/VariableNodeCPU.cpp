#ifdef CPU
#include <memory>

#include "nodes/VariableNode.hpp"
#include "Kernel.hpp"

class VariableNodeCPUKernel : public Kernel
{
public:
    VariableNodeCPUKernel() { }
    ~VariableNodeCPUKernel() { }
    void Run() { }
};

void VariableNode::Compile(GraphCompilationContext& context) const
{
    throw new std::logic_error("not yet implemented.");
}

#endif
