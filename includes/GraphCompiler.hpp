#ifndef _GRAPH_COMPILER_HPP_
#define _GRAPH_COMPILER_CPP_

#include "nodes/Node.hpp"
#include "GraphCompilationContext.hpp"

class GraphCompiler
{
private:
    GraphCompilationContext _context;
public:
    GraphCompiler();
    virtual ~GraphCompiler() { }
};

#endif
