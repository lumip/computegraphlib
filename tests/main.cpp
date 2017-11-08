#include <iostream>
#include <vector>
#include <stdexcept>

#include <CL/cl.h>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "nodes/VectorAddNode.hpp"
#include "nodes/VariableNode.hpp"
#include "GraphCompilationContext.hpp"
#include "GraphCompiler.hpp"

#include "Kernel.hpp"

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
    void Run() const { std::cout << _name << std::endl; }
};

class TestForwardNode : public Node
{
private:
    std::vector<ConstNodePtr> _children;
    std::string _name;
public:
    TestForwardNode(const std::string name, const std::vector<ConstNodePtr>& children) : _name(name), _children(children) {}
    ~TestForwardNode() {}
    ConstNodeList GetInputs() const { return ConstNodeList(_children); }
    void Compile(GraphCompilationContext& context) const { context.EnqueueKernel(std::unique_ptr<const Kernel>(new TestKernel(_name))); }
    std::string ToString() const { return _name; }
    bool IsInitialized() const { return _children.empty(); }
};

int main(int argc, const char* argv[])
{
#ifdef GPU
    auto device = SelectDevice();
    PrintDeviceName(device);
#endif
    TestForwardNode f("f", {});
    TestForwardNode e("e", {&f});
    TestForwardNode d("d", {&e});
    TestForwardNode b("b", {&d});
    TestForwardNode c("c", {&b, &d});
    TestForwardNode h("h", {});
    TestForwardNode g("g", {&h});
    TestForwardNode a("a", {&e, &c, &b, &g});

    // todo: need to get back to the kernels
    /*GraphCompiler compiler;
    const std::vector<std::unique_ptr<const Kernel>> kernels = compiler.Compile(&a, InputDataMap());
    for (size_t i = 0; i < kernels.size(); ++i)
    {
        kernels[i]->Run();
    }*/


    return 0;
}
