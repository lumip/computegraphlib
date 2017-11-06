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

#include <functional>

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

int main(int argc, const char* argv[])
{
#ifdef GPU
    auto device = SelectDevice();
    PrintDeviceName(device);
#endif
    InputNode i1("x", 4);
    InputNode i2("y", 4);
    VariableNode v("v", 4);
    VectorAddNode p1(&i1, &v);
    VectorAddNode p2(&p1, &i2);
    v.SetInput(&p2);
    v.SetInput(nullptr);

    std::vector<ConstNodePtr> nodes {&i1,&i2,&p1,&p2,&v};
    for (auto node : nodes)
    {
        std::cout << node->ToString() << std::endl;
        std::cout << "inputs:" << std::endl;
        for (auto input : node->GetInputs())
        {
            std::cout << "\t" << input.second->ToString() << std::endl;
        }
        std::cout << "subscribers:" << std::endl;
        for (auto sub : node->GetSubscribers())
        {
            std::cout << "\t" << sub->ToString() << std::endl;
        }
    }
    InputDataBuffer input1 { 1,2,3,4, 2,2,2,2, 0,0,0,0, 1,0,-1,0, -1,-3,-5,-7 };
    InputDataBuffer input2 { 4,3,2,1, 2,-2,2,-2, 1,2,3,4, -1,0,1,0, 3,5,7,-3 };
    InputDataBuffer expected { 5,5,5,5, 4,0,4,0, 1,2,3,4, 0,0,0,0, 2,2,2,-10 };

    int a(4);
    std::reference_wrapper<int> a_ref(a);

    InputDataMap inputs;
    inputs.emplace("x", std::ref(input1));
    inputs.emplace("y", std::ref(input2));
    /*GraphCompiler compiler;
    compiler.Compile(nodes, inputs);*/

    VectorAddNode testAddNode(&i1, &i2);
    GraphCompilationContext context(inputs);
    context.AssignNodeMemory(&i1, context.RegisterMemory(5, 4));
    context.AssignNodeMemory(&i2, context.RegisterMemory(5, 4));
    std::unique_ptr<const Kernel> kernel = testAddNode.Compile(&context);
    // todo: have to copy input data into buffers before running
    float* const memInA = context.GetNodeMemoryDescriptor(&i1).handle;
    float* const memInB = context.GetNodeMemoryDescriptor(&i2).handle;
    std::copy(input1.cbegin(), input1.cend(), memInA);
    std::copy(input2.cbegin(), input2.cend(), memInB);
    kernel->Run();

    float* const result = context.GetNodeMemoryDescriptor(&testAddNode).handle;
    float error = 0.0f;
    for (size_t i = 0; i < 20; ++i)
    {
        error += (result[i] - expected[i]) * (result[i] - expected[i]);
    }
    std::cout << "Error: " << error << std::endl;

    return 0;
}
