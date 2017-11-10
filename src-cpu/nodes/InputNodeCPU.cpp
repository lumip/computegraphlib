#ifdef CPU
#include "nodes/InputNode.hpp"
#include "GraphCompilationContext.hpp"
#include "Kernel.hpp"

class InputNodeCPUKernel : public Kernel
{
private:
    InputDataBuffer& _inputBuffer;
    const NodeMemoryDescriptor _mem;
public:
    InputNodeCPUKernel(const NodeMemoryDescriptor workingMemory, InputDataBuffer& inputBuffer)
        : _mem(workingMemory)
        , _inputBuffer(inputBuffer)
    {
    }

    ~InputNodeCPUKernel() { }

    void Run() const
    {
        std::copy(_inputBuffer.cbegin(), _inputBuffer.cend(), reinterpret_cast<float* const>(_mem.handle));
    }
};

void InputNode::Compile(GraphCompilationContext& context) const
{
    InputDataBuffer& input = context.GetInputDataBuffer(_name);
    if (input.size() % this->_dim)
    {
        throw std::invalid_argument("The input buffer size is not a multiple of the dimension size.");
    }
    size_t n = input.size() / this->_dim;
    context.AssignNodeMemory(this, context.RegisterMemory(n, this->_dim));
    const NodeMemoryDescriptor mem = context.GetNodeMemoryDescriptor(this);
    context.RegisterInputMemory(_name, mem.handle);
    context.EnqueueKernel(std::unique_ptr<const Kernel>(new InputNodeCPUKernel(mem, input))); // std::make_unique only since c++14
}

#endif