#ifndef _KERNEL_HPP_
#define _KERNEL_HPP_

#include "types.hpp"

class Kernel
{
public:
    Kernel() { } // todo: disable copy constructor and copy assignment, enable move constructor and move assignment
    virtual ~Kernel() { }
    virtual void Run() = 0;
};

#endif
