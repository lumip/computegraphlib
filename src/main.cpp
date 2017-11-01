#include <iostream>
#include <vector>
#include <stdexcept>

#include <CL/cl.h>

#include "nodes/InputNode.hpp"
#include "nodes/VectorAddNode.hpp"
#include "GraphCompilationContext.hpp"

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
    InputNode i1(4);
    InputNode i2(4);
    VectorAddNode an(&i1, &i2);
    for (auto inputPair : an.GetInputs())
    {
        std::cout << inputPair.second << std::endl;
    }
    GraphCompilationContext context;
    context.RegisterNodeMemory(&i1, 15, 4);
    context.RegisterNodeMemory(&i2, 15, 4);
    context.RegisterNodeMemory(&an, 15, 4);
    
    return 0;
}
