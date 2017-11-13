#ifndef _COMPILED_GRAPH_HPP_
#define _COMPILED_GRAPH_HPP_

#include <string>
#include "types.hpp"

class CompiledGraph
{
public:
    CompiledGraph() { }
    virtual ~CompiledGraph() { }
    virtual void Evaluate(const InputDataMap& inputData) = 0;
    virtual void GetOutputData(std::string outputName, DataBuffer& outputBuffer) const = 0;
    virtual void GetNodeData(const ConstNodePtr node, DataBuffer& outputBuffer) const = 0;
    virtual void SetNodeData(const ConstNodePtr node, InputDataBuffer& inputBuffer) = 0;
};

#endif
