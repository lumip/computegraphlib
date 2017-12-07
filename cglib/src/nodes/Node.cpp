#include "nodes/Node.hpp"

Node::Node()
    : Node(false, false)
{ }

Node::Node(bool canOperateInPlace)
    : Node(canOperateInPlace, false)
{ }

Node::Node(bool canOperateInPlace, bool isInitialized)
    : _canOperateInPlace(canOperateInPlace), _isInitialized(isInitialized), _subscribers()
{ }

Node::~Node() { }

Node::ConstNodeList Node::GetSubscribers() const
{
    return ConstNodeList(std::begin(_subscribers), std::end(_subscribers));
}

void Node::InternalAddSubscriber(const ConstNodePtr sub)
{
    _subscribers.insert(sub);
}

void Node::InternalRemoveSubscriber(const ConstNodePtr sub)
{
    auto it = _subscribers.find(sub);
    if (it != _subscribers.end())
    {
        _subscribers.erase(it);
    }
}

void Node::SubscribeTo(const NodePtr node) const
{
    node->InternalAddSubscriber(this);
}

void Node::UnsubscribeFrom(const NodePtr node) const
{
    node->InternalRemoveSubscriber(this);
}

bool Node::CanOperateInPlace() const
{
    return _canOperateInPlace;
}

bool Node::IsInitialized() const
{
    return _isInitialized;
}
