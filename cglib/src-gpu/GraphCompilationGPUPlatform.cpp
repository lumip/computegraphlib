#include "GraphCompilationGPUPlatform.hpp"

#include <sstream>
#include <set>

#include "CompilationMemoryMap.hpp"
#include "Kernel.hpp"

GraphCompilationGPUPlatform::GraphCompilationGPUPlatform(const CompilationMemoryMap& compilationMemoryMap, OCLWrappers::Context&& context)
    : GraphCompilationPlatform(compilationMemoryMap)
    , _clContext(std::move(context))
    , _clDevice(SelectDevice())
    , _clMemoryQueue(CreateCommandQueue())
    , _clExecutionQueue(CreateCommandQueue())
    , _computeUnitCount(QueryComputeUnitCount())
    , _kernels()
    , _memoryBufferLocations()
    , _clPrograms()
    , _clKernels()
    , _executionFinishedEvent()
    , _isRunning(false)
    , _nodeKernels()
    , _mappedInputBuffers()
{
}

GraphCompilationGPUPlatform::~GraphCompilationGPUPlatform()
{
    clFinish(_clExecutionQueue.get());
    for (auto& elem : _mappedInputBuffers)
    {
        OCLWrappers::Memory& devMem = elem.second.first;
        float* hostMem = elem.second.second;
        OCLWrappers::CheckCLError(
            clEnqueueUnmapMemObject(_clMemoryQueue.get(), devMem.get(), static_cast<void*>(hostMem), 0, nullptr, nullptr)
        , "clEnqueueUnmapMemObject (for ~GraphCompilationGPUPlatform)");
    }
    clFinish(_clMemoryQueue.get());
}

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

size_t GraphCompilationGPUPlatform::QueryComputeUnitCount()
{
    cl_uint computeUnits = 0;
    OCLWrappers::CheckCLError(
        clGetDeviceInfo(_clDevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &computeUnits, nullptr)
    , "clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS) (for SelectDevice)");
    return computeUnits;
}

OCLWrappers::Queue GraphCompilationGPUPlatform::CreateCommandQueue()
{
    cl_int status = CL_SUCCESS;
    cl_queue_properties queueProperties[] =
    {
        CL_QUEUE_PROPERTIES,
        (cl_queue_properties)CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0
    };
    OCLWrappers::Queue queue = clCreateCommandQueueWithProperties(_clContext.get(), _clDevice, queueProperties, &status);
    OCLWrappers::CheckCLError(status, "clCreateCommandQueue");
    return queue;
}

GraphCompilationGPUPlatform::MemoryHandle GraphCompilationGPUPlatform::GetMemoryLocation(const ConstNodePtr node) const
{
    return _memoryBufferLocations[GetNodeMemoryBuffer(node)].get();
}

void GraphCompilationGPUPlatform::AllocateMappedMemory(std::string const& inputName, ConstNodePtr const node)
{
    MemoryDimensions const dims = _dimensionsMap.GetNodeMemoryDimensions(node);
    cl_int status = CL_SUCCESS;
    OCLWrappers::Memory deviceMem = clCreateBuffer(_clContext.get(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, dims.size() * sizeof(float), nullptr, &status);
    OCLWrappers::CheckCLError(status, "clCreateBuffer(CL_MEM_ALLOC_HOST_PTR) for AllocateAllMemory to allocate pinned memory");
    float* hostMem = static_cast<float*>(clEnqueueMapBuffer(_clMemoryQueue.get(), deviceMem.get(), CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, 0, dims.size() * sizeof(float), 0, nullptr, nullptr, &status));
    OCLWrappers::CheckCLError(status, "clEnqueueMapMemory for AllocateAllMemory to map pinned memory.");
    _mappedInputBuffers[inputName] = std::pair<OCLWrappers::Memory, float*>(std::move(deviceMem), hostMem);
}

void GraphCompilationGPUPlatform::AllocateAllMemory()
{
    // allocate device memory buffers for all nodes
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

    // allocate (hopefully) pinned memory for inputs and variables
    for (std::string inputName : _dimensionsMap.GetInputNames())
    {
        ConstNodePtr node = _dimensionsMap.GetInputNode(inputName);
        AllocateMappedMemory(inputName, node);
    }
    for (std::string variableName : _dimensionsMap.GetVariableNames())
    {
        ConstNodePtr node = _dimensionsMap.GetVariableNode(variableName);
        AllocateMappedMemory(variableName, node);
    }
    clFinish(_clMemoryQueue.get());
}

void GraphCompilationGPUPlatform::CopyOutputData(const ConstNodePtr outputNode, float* outputBuffer) const
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(outputNode);
    size_t size = dims.size();
    const cl_mem nodeMemBuffer = GetMemoryLocation(outputNode);
    clFinish(_clExecutionQueue.get());
    OCLWrappers::CheckCLError(
        clEnqueueReadBuffer(_clMemoryQueue.get(), nodeMemBuffer, CL_TRUE, 0, size * sizeof(float), outputBuffer, 0, nullptr, nullptr)
    );
}

void GraphCompilationGPUPlatform::CopyInputData(const ConstNodePtr inputNode, float const* inputBuffer)
{
    MemoryDimensions dims = _dimensionsMap.GetNodeMemoryDimensions(inputNode);
    const cl_mem nodeMemBuffer = GetMemoryLocation(inputNode);
    clFinish(_clExecutionQueue.get());
    OCLWrappers::CheckCLError(
        clEnqueueWriteBuffer(_clMemoryQueue.get(), nodeMemBuffer, CL_FALSE, 0, dims.size() * sizeof(float), inputBuffer, 0, nullptr, nullptr)
    , "clEnqueueWriteBuffer (for CopyInputData)");
}

void GraphCompilationGPUPlatform::Evaluate()
{
    clFinish(_clMemoryQueue.get());
    clFinish(_clExecutionQueue.get()); // make sure that all memory copy and previous executions have finished
    _isRunning = true;
    for (const std::unique_ptr<Kernel>& kernel : _kernels)
    {
        kernel->Run();
    }
    cl_event rawEvent;
    OCLWrappers::CheckCLError(
        clEnqueueMarkerWithWaitList(_clExecutionQueue.get(), 0, nullptr, &rawEvent)
    , "clEnqueueMarkerWithWaitList");
    _executionFinishedEvent = OCLWrappers::Event(rawEvent);
    OCLWrappers::CheckCLError( // set callback to set _isRunning to false. would rather retrieve completion status from event but that does not seem to work..
        clSetEventCallback(rawEvent, CL_COMPLETE,
            [](cl_event, cl_int, void* isRunning){*reinterpret_cast<std::atomic<bool>*>(isRunning) = false;}, &(this->_isRunning))
    , "clSetEventCallback");
    // note: OpenCL docs state that callback might occur asynchronously and should be thread-safe.
}

bool GraphCompilationGPUPlatform::IsEvaluating() const
{
    return _isRunning;
    // the clGetEventInfo method apparently just returns generic error "Out of resources" after evaluation completes
    // so it is not a reliable information source here... have to resort to status variable and completion callback
    /*if (!_executionFinishedEvent.valid())
    {
        return false;
    }
    cl_int eventStatus;
    OCLWrappers::CheckCLError(
        clGetEventInfo(_executionFinishedEvent.get(), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(eventStatus), &eventStatus, nullptr)
    , "clGetEventInfo");
    return !(eventStatus == CL_COMPLETE);*/
}

void GraphCompilationGPUPlatform::WaitUntilEvaluationFinished() const
{
    cl_event eventList[1] = { _executionFinishedEvent.get() };
    OCLWrappers::CheckCLError(clWaitForEvents(1, eventList), "clWaitForEvents");
    while (_isRunning) {} // have to wait until callback occured so that IsEvaluating gives consistent result.....
}

void GraphCompilationGPUPlatform::WaitUntilDataTransferFinished() const
{
    clFinish(_clMemoryQueue.get());
}

float* GraphCompilationGPUPlatform::GetMappedInputBuffer(std::string const& inputName)
{
    return _mappedInputBuffers.at(inputName).second;
}

cl_kernel GraphCompilationGPUPlatform::CompileKernel(const std::string& kernelSource)
{
    if (_clKernels.find(kernelSource) != _clKernels.end())
    {
        return _clKernels[kernelSource].get();
    }
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
    const cl_kernel clKernel = kernel.get();
    _clKernels[kernelSource] = std::move(kernel);
    _clPrograms.push_back(std::move(program)); // if all checks go well, store handle of program. if not, it will be released automatically upon method exit
    return clKernel;
}

size_t GraphCompilationGPUPlatform::GetComputeUnitCount() const
{
    return _computeUnitCount;
}

GPUKernel const* GraphCompilationGPUPlatform::GetNodeKernel(ConstNodePtr node) const
{
    return _nodeKernels.at(node);
}
