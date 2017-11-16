#include "GraphCompilationGPUStrategy.hpp"
#include <sstream>

std::string ErrorCodeToMessage(cl_int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default: return "Unknown";
    }
}

void GraphCompilationGPUStrategy::CheckCLError(cl_int status, const std::string& methodName)
{
    if (status != CL_SUCCESS)
    {
        std::stringstream ss;
        ss << methodName << " failed with error: " << status << " " << ErrorCodeToMessage(status);
        throw std::runtime_error(ss.str());
    }
}

void GraphCompilationGPUStrategy::CheckCLError(cl_int status)
{
    CheckCLError(status, "OpenCL call");
}

GraphCompilationGPUStrategy::GraphCompilationGPUStrategy(const cl_context context)
    : _clContext(context)
    , _clDevice(SelectDevice())
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

cl_device_id GraphCompilationGPUStrategy::SelectDevice()
{
    size_t contextDeviceCount = 0;
    CheckCLError(clGetContextInfo(_clContext, CL_CONTEXT_DEVICES, 0, nullptr, &contextDeviceCount), "clGetContextInfo");
    if (contextDeviceCount == 0)
    {
        throw std::invalid_argument("Given context does not contain devices.");
    }
    std::vector<cl_device_id> devices(contextDeviceCount);
    CheckCLError(clGetContextInfo(_clContext, CL_CONTEXT_DEVICES, contextDeviceCount, devices.data(), nullptr), "clGetContextInfo");
    return devices[0];
}

cl_command_queue GraphCompilationGPUStrategy::CreateCommandQueue()
{
    cl_int status = CL_SUCCESS;
    cl_command_queue queue = clCreateCommandQueue(_clContext, _clDevice, 0/*CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/, &status);
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
        clEnqueueReadBuffer(_clMemoryQueue, reinterpret_cast<cl_mem>(outputNodeMemory), CL_TRUE, 0, size * sizeof(float), outputBuffer.data(), 0, nullptr, nullptr) // todo: temporarily (?) blocking
    );
    clFinish(_clMemoryQueue);
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
    clFinish(_clMemoryQueue);
    for (const std::unique_ptr<Kernel>& kernel : _kernels)
    {
        kernel->Run();
    }
}

cl_kernel GraphCompilationGPUStrategy::CompileKernel(const std::string& kernelSource)
{
    size_t length = kernelSource.length();
    const char* s = kernelSource.c_str();
    const char** sources = &s;
    cl_int status = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(_clContext, 1, sources, &length, &status);
    CheckCLError(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (status == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t buildLogSize = 0;
        CheckCLError(clGetProgramBuildInfo(program, _clDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildLogSize), "clGetProgramBuildInfo");
        char* const buildLog = new char[buildLogSize];
        CheckCLError(clGetProgramBuildInfo(program, _clDevice, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, nullptr), "clGetProgramBuildInfo");
        std::stringstream ss;
        ss << "clBuildProgram: GPU kernel did not compile! Error Message:\n" << buildLog;
        delete[] buildLog;
        throw std::runtime_error(ss.str());
    }
    else
    {
        CheckCLError(status, "clBuildProgram");
    }
    cl_kernel kernel = clCreateKernel(program, "main", &status);
    CheckCLError(status, "clCreateKernel");
    return kernel;
}
