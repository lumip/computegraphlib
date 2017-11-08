#ifndef _NODE_HPP_
#define _NODE_HPP_

#include <memory>
#include <map>
#include <vector>
#include <set>

#include "Kernel.hpp"
#include "types.hpp"

class GraphCompilationContext;

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
    ConstNodeSet _subscribers;
public:
    Node();
    virtual ~Node();
    ConstNodeList GetSubscribers() const;
    virtual ConstNodeList GetInputs() const = 0;
    virtual void Compile(GraphCompilationContext& context) const = 0;
    virtual std::string ToString() const = 0;
    virtual bool IsInitialized() const = 0;
private:
    void InternalAddSubscriber(const ConstNodePtr sub);
    void InternalRemoveSubscriber(const ConstNodePtr sub);
protected:
    void SubscribeTo(const NodePtr node) const;
    void UnsubscribeFrom(const NodePtr node) const;
};

#endif
