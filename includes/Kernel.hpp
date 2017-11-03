#ifndef _KERNEL_HPP_
#define _KERNEL_HPP_

class Kernel
{
public:
    Kernel() { }
    virtual ~Kernel() { }
    virtual void Run() const = 0;
};

#endif
