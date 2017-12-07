#ifndef _NODE_HPP_
#define _NODE_HPP_

#include <memory>
#include <map>
#include <vector>
#include <set>

#include "Kernel.hpp"
#include "types.hpp"

class CompilationMemoryMap;
class GraphCompilationPlatform;

class Node
{
protected:
    typedef std::set<NodePtr> NodeSet;
    typedef std::vector<NodePtr> NodeList;
    typedef std::map<std::string, NodePtr> NodeMap;
public:
    typedef std::set<ConstNodePtr> ConstNodeSet;
    typedef std::map<std::string, ConstNodePtr> ConstNodeMap;
    typedef std::vector<ConstNodePtr> ConstNodeList;
private:
    const bool _canOperateInPlace;
    const bool _isInitialized;
    ConstNodeSet _subscribers;    
public:
    Node();
    Node(bool canOperateInPlace);
    Node(bool canOperateInPlace, bool isInitialized);
    virtual ~Node();

    Node(const Node&) = delete;
    Node(Node&&) = delete;
    Node& operator=(const Node&) = delete;
    Node& operator=(Node&&) = delete;

    ConstNodeList GetSubscribers() const;
    virtual ConstNodeList GetInputs() const = 0;
    virtual void Compile(GraphCompilationPlatform& platform) const = 0;
    virtual void GetMemoryDimensions(CompilationMemoryMap& memoryMap) const = 0;
    virtual std::string ToString() const = 0;
    virtual bool IsInitialized() const;
    virtual bool CanOperateInPlace() const;
private:
    void InternalAddSubscriber(const ConstNodePtr sub);
    void InternalRemoveSubscriber(const ConstNodePtr sub);
protected:
    void SubscribeTo(const NodePtr node) const;
    void UnsubscribeFrom(const NodePtr node) const;
};

#endif
