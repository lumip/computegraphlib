#ifndef _INPUT_NODE_HPP_
#define _INPUT_NODE_HPP_

#include "Node.hpp"

class InputNode : public Node
{
public:
    typedef std::weak_ptr<InputNode> weak_ptr;
    typedef std::shared_ptr<InputNode> shared_ptr;
    typedef std::unique_ptr<InputNode> unique_ptr;
private:
    std::string _name;
    size_t _dim;
public:
    InputNode(std::string name, size_t dim);
    virtual ~InputNode();
    ConstNodeMap GetInputs() const;
    void Compile(GraphCompilationContext* const context) const;
    std::string ToString() const;
};

#endif
