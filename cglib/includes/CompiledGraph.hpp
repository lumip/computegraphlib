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
    void Evaluate(const InputDataMap& inputData);
    bool IsEvaluating() const;
    void WaitUntilFinished() const;
    void InitializeVariables(const InputDataMap& inputData);
    void GetNodeData(const ConstNodePtr node, float* outputBuffer) const;
    void SetNodeData(const ConstNodePtr node, float const* inputBuffer);
    float* GetMappedInputBuffer(std::string const& inputName);
};

#endif
