#include "nodes/Node.hpp"

Node::Node()
    : _subscribers()
{

}

Node::~Node() { }

Node::ConstNodeList Node::GetSubscribers() const
{
    return ConstNodeList(std::begin(_subscribers), std::end(_subscribers));
}

void Node::InternalAddSubscriber(const Node::const_ptr sub)
{
    _subscribers.insert(sub);
}

void Node::InternalRemoveSubscriber(const Node::const_ptr sub)
{
    auto it = _subscribers.find(sub);
    if (it != _subscribers.end())
    {
        _subscribers.erase(it);
    }
}

void Node::SubscribeTo(const Node::ptr node) const
{
    node->InternalAddSubscriber(this);
}

void Node::UnsubscribeFrom(const Node::ptr node) const
{
    node->InternalRemoveSubscriber(this);
}
