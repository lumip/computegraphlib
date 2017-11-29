#ifndef _SLICE_NODE_HPP_
#define _SLICE_NODE_HPP_

#include "nodes/Node.hpp"

class SliceNode : public Node
{
private:
    const ConstNodePtr _input;
    const size_t _slice_id;
    const size_t _axis;
public:
    SliceNode(NodePtr input, size_t slice_id, size_t axis);
    virtual ~SliceNode();
    ConstNodeList GetInputs() const;
    void Compile(GraphCompilationPlatform& platform) const;
    void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const;
    std::string ToString() const;
    bool IsInitialized() const;
};

#endif
