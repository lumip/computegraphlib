#include <iostream>
#include <vector>
#include <stdexcept>

#include <CL/cl.h>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "nodes/VectorAddNode.hpp"
#include "GraphCompilationContext.hpp"
#include "GraphCompiler.hpp"

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
    VectorAddNode an(&i1, &i2);
    for (auto inputPair : an.GetInputs())
    {
        std::cout << inputPair.second->ToString() << std::endl;
    }
    for (auto x : i1.GetSubscribers())
    {
        std::cout << x->ToString() << std::endl;
    }

    std::vector<Node::const_ptr> nodes;
    nodes.push_back(&i1);
    nodes.push_back(&i2);
    nodes.push_back(&an);
    float input1[5][4] = { {1,2,3,4}, {2,2,2,2}, {0,0,0,0}, {1,0,-1,0}, {-1,-3,-5,-7} };
    float input2[5][4] = { {4,3,2,1}, {2,-2,2,-2}, {1,2,3,4}, {-1,0,1,0}, {3, 5, 7, -3} };
    InputDataMap inputs;
    inputs["x"] = static_cast<InputDataBufferHandle>(&input1[0][0]);
    inputs["y"] = static_cast<InputDataBufferHandle>(&input2[0][0]);
    GraphCompiler compiler;
    compiler.Compile(nodes, inputs);
    
    return 0;
}
