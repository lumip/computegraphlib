#ifndef _NODE_HPP_
#define _NODE_HPP_

#include <memory>
#include <map>
#include <vector>
#include <set>

class Node
{
public:
    typedef std::weak_ptr<Node> weak_ptr;
    typedef std::shared_ptr<Node> shared_ptr;
    typedef std::unique_ptr<Node> unique_ptr;
    typedef Node* ptr;
    typedef const Node* const_ptr;
protected:
    typedef std::set<Node::ptr> NodeSet;
    typedef std::vector<Node::ptr> NodeList;
    typedef std::map<std::string, Node::ptr> NodeMap;
public:
    typedef std::set<Node::const_ptr> ConstNodeSet;
    typedef std::map<std::string, Node::const_ptr> ConstNodeMap;
    typedef std::vector<Node::const_ptr> ConstNodeList;
private:
    ConstNodeSet _subscribers;
public:
    Node() {}
    virtual ~Node() {}
    virtual ConstNodeMap GetInputs() const = 0;
    virtual void Compile(void* const context) const = 0;
};

#endif
