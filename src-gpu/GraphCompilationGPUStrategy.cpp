#ifdef GPU

#include "GraphCompilationGPUStrategy.hpp"
#include <sstream>

void GraphCompilationGPUStrategy::CheckCLError(cl_int status, const std::string& methodName)
{
    if (status != CL_SUCCESS)
    {
        std::stringstream ss;
        ss << methodName << " failed with error: " << status;
        throw std::runtime_error(ss.str());
    }
}

void GraphCompilationGPUStrategy::CheckCLError(cl_int status)
{
    CheckCLError(status, "OpenCL call");
}

GraphCompilationGPUStrategy::GraphCompilationGPUStrategy(const cl_context context)
    : _clContext(context)
    , _clMemoryQueue(CreateCommandQueue())
    , _clExecutionQueue(CreateCommandQueue())
    , _kernels()
{
}

GraphCompilationGPUStrategy::~GraphCompilationGPUStrategy()
{
    clReleaseCommandQueue(_clMemoryQueue);
    clReleaseCommandQueue(_clExecutionQueue);
    clReleaseContext(_clContext);
}

cl_command_queue GraphCompilationGPUStrategy::CreateCommandQueue()
{
    size_t contextDeviceCount = 0;
    CheckCLError(clGetContextInfo(_clContext, CL_CONTEXT_DEVICES, 0, nullptr, &contextDeviceCount), "clGetContextInfo");
    if (contextDeviceCount == 0)
    {
        throw std::invalid_argument("Given context does not contain devices.");
    }
    std::vector<cl_device_id> devices(contextDeviceCount);
    CheckCLError(clGetContextInfo(_clContext, CL_CONTEXT_DEVICES, contextDeviceCount, devices.data(), nullptr), "clGetContextInfo");
    cl_int status = CL_SUCCESS;
    cl_command_queue queue = clCreateCommandQueue(_clContext, devices[0], 0/*CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/, &status);
    CheckCLError(status, "clCreateCommandQueue");
    return queue;
}

NodeMemoryHandle GraphCompilationGPUStrategy::AllocateMemory(size_t size)
{
    cl_int status = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(_clContext, CL_MEM_READ_WRITE, size * sizeof(float), nullptr, &status);
    CheckCLError(status, "clCreateBuffer");
    return reinterpret_cast<NodeMemoryHandle>(buffer);
}

void GraphCompilationGPUStrategy::DeallocateMemory(const NodeMemoryHandle mem)
{
    cl_mem buffer = reinterpret_cast<cl_mem>(mem);
    CheckCLError(clReleaseMemObject(buffer), "clReleaseMemObject");
}

void GraphCompilationGPUStrategy::EnqueueKernel(std::unique_ptr<Kernel>&& kernel)
{
    _kernels.emplace_back(std::move(kernel));
}

void GraphCompilationGPUStrategy::CopyOutputData(const NodeMemoryHandle outputNodeMemory, DataBuffer& outputBuffer, size_t size) const
{
    clFinish(_clExecutionQueue);
    CheckCLError(
        clEnqueueReadBuffer(_clMemoryQueue, reinterpret_cast<cl_mem>(outputNodeMemory), CL_FALSE, 0, size * sizeof(float), outputBuffer.data(), 0, nullptr, nullptr)
    );
}

void GraphCompilationGPUStrategy::CopyInputData(const NodeMemoryHandle inputNodeMemory, InputDataBuffer& inputBuffer, size_t size)
{
    clFinish(_clExecutionQueue);
    CheckCLError(
        clEnqueueWriteBuffer(_clMemoryQueue, reinterpret_cast<cl_mem>(inputNodeMemory), CL_FALSE, 0, size * sizeof(float), inputBuffer.data(), 0, nullptr, nullptr) // todo: handle events?
    );
}

void GraphCompilationGPUStrategy::Evaluate(const std::vector<std::pair<const NodeMemoryDescriptor, InputDataBuffer&>>& inputData)
{
    clFinish(_clMemoryQueue);
    clFinish(_clExecutionQueue);
    for (auto input : inputData)
    {
         CopyInputData(input.first.handle, input.second, input.first.dimensions.size());
    }
    for (const std::unique_ptr<Kernel>& kernel : _kernels)
    {
        kernel->Run();
    }
}

cl_program GraphCompilationGPUStrategy::CompileKernel(const std::string& kernelSource)
{
    size_t length = kernelSource.length();
    const char* s = kernelSource.c_str();
    const char** sources = &s;
    cl_int status = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(_clContext, 1, sources, &length, &status);
    CheckCLError(status, "clCreateProgramWithSource");
    CheckCLError(
        clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr)
    , "clBuildprogram");
    return program;
}

#endif
