#include "GraphCompilationGPUPlatform.hpp"

#include <sstream>
#include <set>

#include "CompilationMemoryMap.hpp"
#include "Kernel.hpp"

struct GraphCompilationGPUPlatform::MemoryBuffer
{
    MemoryDimensions Dimensions;
    std::set<ConstNodePtr> Subscribers;
};

GraphCompilationGPUPlatform::GraphCompilationGPUPlatform(const CompilationMemoryMap& CompilationMemoryMap, OCLWrappers::Context&& context)
    : _dimensionsMap(CompilationMemoryMap)
    , _clContext(std::move(context))
    , _clDevice(SelectDevice())
    , _clMemoryQueue(CreateCommandQueue())
    , _clExecutionQueue(CreateCommandQueue())
    , _kernels()
    , _memoryBufferLocations()
    , _memoryBuffers()
    , _nodeMemoryAssignments()
{
}

GraphCompilationGPUPlatform::~GraphCompilationGPUPlatform() { }

cl_device_id GraphCompilationGPUPlatform::SelectDevice()
{
    size_t contextDeviceCount = 0;
    OCLWrappers::CheckCLError(clGetContextInfo(_clContext.get(), CL_CONTEXT_DEVICES, 0, nullptr, &contextDeviceCount), "clGetContextInfo");
    if (contextDeviceCount == 0)
    {
        throw std::invalid_argument("Given context does not contain devices.");
    }
    std::vector<cl_device_id> devices(contextDeviceCount);
    OCLWrappers::CheckCLError(clGetContextInfo(_clContext.get(), CL_CONTEXT_DEVICES, contextDeviceCount, devices.data(), nullptr), "clGetContextInfo");
    return devices[0];
}

OCLWrappers::Queue GraphCompilationGPUPlatform::CreateCommandQueue()
{
    cl_int status = CL_SUCCESS;
    // todo: set cl_queue_properties
    OCLWrappers::Queue queue = clCreateCommandQueueWithProperties(_clContext.get(), _clDevice, nullptr, &status);
    OCLWrappers::CheckCLError(status, "clCreateCommandQueue");
    return queue;
}

GraphCompilationGPUPlatform::MemoryHandle GraphCompilationGPUPlatform::GetMemoryLocation(const ConstNodePtr node) const
{
    return _memoryBufferLocations[_nodeMemoryAssignments.at(node)].get();
}

GraphCompilationPlatform::MemoryBufferHandle GraphCompilationGPUPlatform::ReserveMemoryBuffer(const ConstNodePtr node)
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    size_t size = dims.size();
    if (size == 0)
    {
        std::invalid_argument("cannot allocate memory of size 0!");
    }
    MemoryBufferHandle handle = static_cast<MemoryBufferHandle>(_memoryBuffers.size());
    MemoryBuffer buffer;
    buffer.Dimensions = dims;
    buffer.Subscribers.insert(node);
    _memoryBuffers.push_back(buffer);
    AssignMemoryBuffer(node, handle);
    return handle;
}

void GraphCompilationGPUPlatform::AssignMemoryBuffer(const ConstNodePtr node, MemoryBufferHandle memory)
{
    if (NodeIsAssigned(node))
    {
        // if the node is already assigned to a memory handle, we will have two merge the old and the new buffers:
        // - move all subscribers from the previous to the new buffer and clear the previous buffer (abandon it)
        MemoryBuffer& oldBuffer = _memoryBuffers[GetNodeMemoryBuffer(node)];
        MemoryBuffer& newBuffer = _memoryBuffers[memory];
        newBuffer.Subscribers.insert(oldBuffer.Subscribers.cbegin(), oldBuffer.Subscribers.cend());
        for (ConstNodePtr n : oldBuffer.Subscribers)
        {
            _nodeMemoryAssignments[n] = memory;
        }
        oldBuffer.Subscribers.clear();
        oldBuffer.Dimensions = {0, 0};
    }
    else
    {
        _nodeMemoryAssignments[node] = memory;
        _memoryBuffers[memory].Subscribers.insert(node);
    }
}

GraphCompilationPlatform::MemoryBufferHandle GraphCompilationGPUPlatform::GetNodeMemoryBuffer(const ConstNodePtr node) const
{
    return _nodeMemoryAssignments.at(node);
}

bool GraphCompilationGPUPlatform::NodeIsAssigned(const ConstNodePtr node) const
{
    return _nodeMemoryAssignments.find(node) != _nodeMemoryAssignments.end();
}

void GraphCompilationGPUPlatform::AllocateAllMemory()
{
    _memoryBufferLocations.resize(_memoryBuffers.size());
    for (size_t i = 0; i < _memoryBuffers.size(); ++i)
    {
        const MemoryBuffer& buffer = _memoryBuffers[i];
        if (buffer.Subscribers.size() > 0)
        {
            cl_int status = CL_SUCCESS;
            OCLWrappers::Memory mem = clCreateBuffer(_clContext.get(), CL_MEM_READ_WRITE, buffer.Dimensions.size() * sizeof(float), nullptr, &status);
            OCLWrappers::CheckCLError(status, "clCreateBuffer");
            _memoryBufferLocations[i] = std::move(mem);
        }
    }
}

void GraphCompilationGPUPlatform::CopyOutputData(const ConstNodePtr outputNode, DataBuffer& outputBuffer) const
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(outputNode);
    size_t size = dims.size();
    outputBuffer.resize(size);
    const cl_mem nodeMemBuffer = GetMemoryLocation(outputNode);
    clFinish(_clExecutionQueue.get());
    OCLWrappers::CheckCLError(
        clEnqueueReadBuffer(_clMemoryQueue.get(), nodeMemBuffer, CL_TRUE, 0, size * sizeof(float), outputBuffer.data(), 0, nullptr, nullptr) // todo: temporarily (?) blocking
    );
}

void GraphCompilationGPUPlatform::CopyInputData(const ConstNodePtr inputNode, InputDataBuffer& inputBuffer)
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    const cl_mem nodeMemBuffer = GetMemoryLocation(inputNode);
    clFinish(_clExecutionQueue.get());
    OCLWrappers::CheckCLError(
        clEnqueueWriteBuffer(_clMemoryQueue.get(), nodeMemBuffer, CL_FALSE, 0, dims.size() * sizeof(float), inputBuffer.data(), 0, nullptr, nullptr) // todo: handle events?
    , "clEnqueueWriteBuffer (for CopyInputData)");
}

void GraphCompilationGPUPlatform::Evaluate()
{
    clFinish(_clMemoryQueue.get());
    for (const std::unique_ptr<Kernel>& kernel : _kernels)
    {
        kernel->Run();
    }
}

OCLWrappers::Kernel GraphCompilationGPUPlatform::CompileKernel(const std::string& kernelSource)
{
    size_t length = kernelSource.length();
    const char* s = kernelSource.c_str();
    const char** sources = &s;
    cl_int status = CL_SUCCESS;
    OCLWrappers::Program program = clCreateProgramWithSource(_clContext.get(), 1, sources, &length, &status);
    OCLWrappers::CheckCLError(status, "clCreateProgramWithSource");
    status = clBuildProgram(program.get(), 0, nullptr, nullptr, nullptr, nullptr);
    if (status == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t buildLogSize = 0;
        OCLWrappers::CheckCLError(clGetProgramBuildInfo(program.get(), _clDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &buildLogSize), "clGetProgramBuildInfo");
        std::vector<char> buildLog(buildLogSize);
        OCLWrappers::CheckCLError(clGetProgramBuildInfo(program.get(), _clDevice, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog.data(), nullptr), "clGetProgramBuildInfo");
        std::stringstream ss;
        ss << "clBuildProgram: GPU kernel did not compile! Error Message:\n" << buildLog.data();
        throw std::runtime_error(ss.str());
    }
    else
    {
        OCLWrappers::CheckCLError(status, "clBuildProgram");
    }
    OCLWrappers::Kernel kernel = clCreateKernel(program.get(), "main", &status);
    OCLWrappers::CheckCLError(status, "clCreateKernel");
    _programs.emplace_back(std::move(program)); // if all checks go well, store handle of program. if not, it will be released automatically upon method exit
    return kernel;
}
