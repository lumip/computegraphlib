#ifdef CPU
#include "nodes/InputNode.hpp"
#include "GraphCompilationContext.hpp"
#include "Kernel.hpp"

class InputNodeCPUKernel : public Kernel
{
private:
    //InputDataBuffer& _inputBuffer;
    const NodeMemoryDescriptor _mem;
public:
    InputNodeCPUKernel(const NodeMemoryDescriptor workingMemory/*, InputDataBuffer& inputBuffer*/)
        : _mem(workingMemory)
        //, _inputBuffer(inputBuffer)
    {
    }

    ~InputNodeCPUKernel() { }

    void Run() const
    {
        throw std::logic_error("not correctly implemented");
        //std::copy(_inputBuffer.cbegin(), _inputBuffer.cend(), reinterpret_cast<float* const>(_mem.handle));
    }
};

void InputNode::Compile(GraphCompilationContext& context) const
{
    const MemoryDimensions dim = context.GetInputDimensions(_name);
    if (dim.xDim != this->_dim)
    {
        throw std::invalid_argument("Dimension size for input " + _name + "differ between node declaration and input map declaration.");
    }
    const NodeMemoryDescriptor memDesc = context.RegisterMemory(dim);
    context.AssignNodeMemory(this, memDesc.handle);
    context.RegisterInputMemory(_name, memDesc.handle);
    context.EnqueueKernel(std::unique_ptr<const Kernel>(new InputNodeCPUKernel(memDesc/*, input*/))); // std::make_unique only since c++14
}

#endif
