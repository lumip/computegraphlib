#ifndef _COMPILED_GRAPH_HPP_
#define _COMPILED_GRAPH_HPP_

#include <memory>
#include <string>
#include "types.hpp"

class GraphCompilationPlatform;
class MemoryCompilationMap;

class CompiledGraph
{
private:
    std::unique_ptr<GraphCompilationPlatform> _platform;
    std::unique_ptr<MemoryCompilationMap> _memoryCompilationMap;
public:
    CompiledGraph(std::unique_ptr<GraphCompilationPlatform>&& platform, std::unique_ptr<MemoryCompilationMap>&& memoryCompilationMap);
    virtual ~CompiledGraph();
    //virtual DataBuffer& GetFastInputBuffer(const std::string inputName) = 0; // todo: implement this. gives a buffer for the requested input which enables fast transfer of the data written therein to the kernel (pinned memory for OpenCL, direct buffer access for CPU)
    void Evaluate(const InputDataMap& inputData);
    void GetOutputData(std::string outputName, DataBuffer& outputBuffer) const;
    void GetNodeData(const ConstNodePtr node, DataBuffer& outputBuffer) const;
    void SetNodeData(const ConstNodePtr node, InputDataBuffer& inputBuffer);
};

#endif
