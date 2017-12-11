#ifndef _COMPILED_GRAPH_HPP_
#define _COMPILED_GRAPH_HPP_

#include <memory>
#include <string>

#include "types.hpp"

class GraphCompilationPlatform;
class CompilationMemoryMap;

class CompiledGraph
{
private:
    std::unique_ptr<GraphCompilationPlatform> _platform;
    std::unique_ptr<CompilationMemoryMap> _compilationMemoryMap;
public:
    CompiledGraph(std::unique_ptr<GraphCompilationPlatform>&& platform, std::unique_ptr<CompilationMemoryMap>&& CompilationMemoryMap);
    virtual ~CompiledGraph();
    //virtual float* GetFastInputBuffer(const std::string inputName) = 0; // todo: implement this. gives a buffer for the requested input which enables fast transfer of the data written therein to the kernel (pinned memory for OpenCL, direct buffer access for CPU)
    void Evaluate(const InputDataMap& inputData);
    bool IsEvaluating() const;
    void WaitUntilFinished() const;
    void InitializeVariables(const InputDataMap& inputData);
    void GetNodeData(const ConstNodePtr node, float* outputBuffer) const;
    void SetNodeData(const ConstNodePtr node, float const* inputBuffer);
};

#endif
