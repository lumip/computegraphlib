#ifndef _COMPILED_GRAPH_HPP_
#define _COMPILED_GRAPH_HPP_

#include <string>
#include "types.hpp"

class CompiledGraph
{
public:
    CompiledGraph() { }
    virtual ~CompiledGraph() { }
    //virtual DataBuffer& GetFastInputBuffer(const std::string inputName) = 0; // todo: implement this. gives a buffer for the requested input which enables fast transfer of the data written therein to the kernel (pinned memory for OpenCL, direct buffer access for CPU)
    virtual void Evaluate(const InputDataMap& inputData) = 0;
    virtual void GetOutputData(std::string outputName, DataBuffer& outputBuffer) const = 0;
    virtual void GetNodeData(const ConstNodePtr node, DataBuffer& outputBuffer) const = 0;
    virtual void SetNodeData(const ConstNodePtr node, InputDataBuffer& inputBuffer) = 0;
};

#endif
