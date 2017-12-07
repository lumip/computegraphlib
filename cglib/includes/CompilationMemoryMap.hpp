#ifndef _MEMORY_COMPILATION_MAP_HPP_
#define _MEMORY_COMPILATION_MAP_HPP_

#include <memory>

#include "types.hpp"

class CompilationMemoryMap
{
private:
    InputDimensionsMap _inputDimensions;
    std::map<ConstNodePtr, MemoryDimensions> _memoryMap;
    std::map<std::string, ConstNodePtr> _inputMemoryMap;
public:
    CompilationMemoryMap();
    CompilationMemoryMap(const InputDimensionsMap inputDimensions);
    virtual ~CompilationMemoryMap();
    CompilationMemoryMap(const CompilationMemoryMap& other);
    CompilationMemoryMap(CompilationMemoryMap&& other);
    CompilationMemoryMap& operator=(CompilationMemoryMap other);
    void swap(CompilationMemoryMap& a, CompilationMemoryMap& b);

    void RegisterNodeMemory(const ConstNodePtr node, const MemoryDimensions MemoryDimensions); // todo: add readonly flag
    MemoryDimensions GetNodeMemoryDimensions(const ConstNodePtr node) const;
    MemoryDimensions GetInputDimensions(std::string inputName) const;
    void RegisterInputMemory(const std::string inputName, const ConstNodePtr node);
    ConstNodePtr GetInputNode(const std::string inputName) const;
};

#endif
