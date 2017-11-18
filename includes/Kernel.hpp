#ifndef _KERNEL_HPP_
#define _KERNEL_HPP_

#include "types.hpp"

class Kernel
{
public:
    Kernel() { }
    virtual ~Kernel() { }
    Kernel(const Kernel&) = delete;
    void operator=(const Kernel&) = delete;
    virtual void Run() = 0;
};

#endif
