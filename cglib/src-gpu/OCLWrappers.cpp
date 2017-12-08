#include "OCLWrappers.hpp"

#include <string>
#include <sstream>
#include <stdexcept>

namespace OCLWrappers
{

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

    void CheckCLError(cl_int status, const std::string& methodName)
    {
        if (status != CL_SUCCESS)
        {
            std::stringstream ss;
            ss << methodName << " failed with error: " << status << " " << ErrorCodeToMessage(status);
            throw std::runtime_error(ss.str());
        }
    }

    void CheckCLError(cl_int status)
    {
        CheckCLError(status, "OpenCL call");
    }

    template<typename T, cl_int(*releaseFct)(T)>
    Wrapper<T, releaseFct>::Wrapper() : _value(), _owning(false) { }

    template<typename T, cl_int(*releaseFct)(T)>
    Wrapper<T, releaseFct>::Wrapper(const T value) : _value(value), _owning(true) { }

    template<typename T, cl_int(*releaseFct)(T)>
    Wrapper<T, releaseFct>::~Wrapper()
    {
        if (_owning)
        {
            CheckCLError(
                releaseFct(_value)
            , "releaseFct in Wrapper");
        }
    }

    template<typename T, cl_int(*releaseFct)(T)>
    Wrapper<T, releaseFct>::Wrapper(Wrapper && moved) : _value(moved._value), _owning(true)
    {
        moved._owning = false;
    }

    template<typename T, cl_int(*releaseFct)(T)>
    Wrapper<T, releaseFct>& Wrapper<T, releaseFct>::operator=(Wrapper && moved)
    {
        std::swap(_value, moved._value);
        std::swap(_owning, moved._owning);
        return *this;
    }

    template<typename T, cl_int(*releaseFct)(T)>
    T Wrapper<T, releaseFct>::get() const
    {
        if (_owning)
        {
            return _value;
        }
        throw std::logic_error("call to invalidated wrapper.");
    }

    template<typename T, cl_int(*releaseFct)(T)>
    T Wrapper<T, releaseFct>::operator*() const
    {
        return get();
    }

    template class Wrapper<cl_mem, clReleaseMemObject>;
    template class Wrapper<cl_context, clReleaseContext>;
    template class Wrapper<cl_device_id, clReleaseDevice>;
    template class Wrapper<cl_command_queue, clReleaseCommandQueue>;
    template class Wrapper<cl_kernel, clReleaseKernel>;
    template class Wrapper<cl_program, clReleaseProgram>;



}
