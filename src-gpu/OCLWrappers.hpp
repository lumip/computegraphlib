#ifndef _OCL_WRAPPERS_HPP_
#define _OCL_WRAPPERS_HPP_

#include <CL/cl.h>
#include <iostream>
namespace OCLWrappers
{

    std::string ErrorCodeToMessage(cl_int err);

    void CheckCLError(cl_int status, const std::string& methodName);

    void CheckCLError(cl_int status);

    template<typename T, cl_int(*releaseFct)(T)>
    class Wrapper
    {
    private:
        T _value;
        bool _owning;
    public:
        Wrapper(const T value);
        ~Wrapper();
        Wrapper(Wrapper const &) = delete;
        void operator=(Wrapper const &) = delete;
        Wrapper(Wrapper && moved);
        void operator=(Wrapper &&);
        T get() const;
        T operator*() const;
    };

    typedef Wrapper<cl_mem, clReleaseMemObject> Memory;
    typedef Wrapper<cl_context, clReleaseContext> Context;
    typedef Wrapper<cl_device_id, clReleaseDevice> Device;
    typedef Wrapper<cl_command_queue, clReleaseCommandQueue> Queue;
    typedef Wrapper<cl_kernel, clReleaseKernel> Kernel;
    typedef Wrapper<cl_program, clReleaseProgram> Program;

}

#endif
