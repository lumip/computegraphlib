#ifndef _COMPILED_GRAPH_HPP_
#define _COMPILED_GRAPH_HPP_

#include <string>
#include "types.hpp"

class CompiledGraph
{
public:
    CompiledGraph() { }
    virtual ~CompiledGraph() { }
    virtual void Evaluate() const = 0;
    virtual void GetOutputData(std::string outputName, DataBuffer& outputBuffer) const = 0;
};

#endif
