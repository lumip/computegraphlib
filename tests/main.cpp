#include <iostream>
#include <vector>
#include <stdexcept>

#include <CL/cl.h>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "nodes/VectorAddNode.hpp"
#include "nodes/VariableNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompiler.hpp"
#include "CompiledGraph.hpp"

#include "ImplementationStrategyFactory.hpp"

#include "Kernel.hpp"
#include "../src-gpu/OCLWrappers.hpp"

#ifdef GPU
/**
 * Returns the first device offered by OpenCL.
 */
cl_device_id SelectDevice()
{
    cl_int status = CL_SUCCESS;
    cl_uint platformIdCount = 0;
    status = clGetPlatformIDs(0, nullptr, &platformIdCount);
    if (status != CL_SUCCESS || platformIdCount == 0)
    {
        throw std::runtime_error("No OpenCL platforms available.");
    }
    cl_platform_id platform;
    status = clGetPlatformIDs(1, &platform, nullptr);
    if (status != CL_SUCCESS)
    {
        throw std::runtime_error("Could not obtain platform id.");
    }
    cl_uint deviceIdCount = 0;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);
    if (status != CL_SUCCESS || deviceIdCount == 0)
    {
        throw std::runtime_error("No device available for the OpenCL platform.");
    }
    cl_device_id device;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
    if (status != CL_SUCCESS)
    {
        throw std::runtime_error("Could not obtain device id.");
    }
    return device;
}

void PrintDeviceName(cl_device_id device)
{
    size_t bufferLength = 0;
    clGetDeviceInfo(device, CL_DEVICE_NAME, bufferLength, nullptr, &bufferLength);
    std::cout << bufferLength << std::endl;
    char* buffer = new char[bufferLength];
    clGetDeviceInfo(device, CL_DEVICE_NAME, bufferLength, buffer, nullptr);
    std::cout << buffer << std::endl;
    delete[](buffer);
}
#endif

class TestKernel : public Kernel
{
private:
    std::string _name;
public:
    TestKernel(std::string name) : _name(name) { }
    void Run() { std::cout << _name << std::endl; }
};

class TestForwardNode : public Node
{
private:
    std::string _name;
    bool _isInitialized;
    std::vector<NodePtr> _children;
public:
    TestForwardNode(const std::string name, bool isInitialized, const std::vector<NodePtr>& children) : Node(true, isInitialized), _name(name), _isInitialized(isInitialized), _children() { SetInputs(children); }
    ~TestForwardNode() {}
    ConstNodeList GetInputs() const { return ConstNodeList(std::begin(_children), std::end(_children)); }
    void SetInputs(const std::vector<NodePtr>& inputs)
    {
        for (NodePtr child : _children) { UnsubscribeFrom(child); }
        _children = std::vector<NodePtr>(inputs.cbegin(), inputs.cend());
        for (NodePtr child : _children) { SubscribeTo(child); }
    }
    void Compile(GraphCompilationPlatform& platform) const { std::cout << _name << std::endl; /*context.EnqueueKernel(std::unique_ptr<Kernel>(new TestKernel(_name)));*/ }
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const { memoryMap.RegisterNodeMemory(this, MemoryDimensions({10, 10})); }
    std::string ToString() const { return _name; }
};

int main(int argc, const char* argv[])
{
#ifdef GPU
    OCLWrappers::Device device = SelectDevice();
    PrintDeviceName(*device);
#endif
    /*TestForwardNode h("h", {});
    TestForwardNode g("g", {&h});
    TestForwardNode f("f", {});
    TestForwardNode e("e", {&f});
    TestForwardNode d("d", {&e});
    TestForwardNode b("b", {&d});
    TestForwardNode c("c", {&b, &d});
    TestForwardNode a("a", {&e, &c, &b, &g});*/
    TestForwardNode e("e", true, {});
    TestForwardNode v("v", true, {});
    TestForwardNode c("c", false, {&e, &v});
    TestForwardNode d("d", false, {&e});
    TestForwardNode b("b", false, {&c, &d});
    TestForwardNode a("a", false, {&b});
    TestForwardNode g("g", false, {&b});
    TestForwardNode f("f", false, {&g});
    e.SetInputs({&f});

    GraphCompiler compiler(std::unique_ptr<const ImplementationStrategyFactory>(new ImplementationStrategyFactory));
    const std::unique_ptr<CompiledGraph> graph = compiler.Compile(&a, InputDimensionsMap());
    graph->Evaluate(InputDataMap());

    return 0;
}
