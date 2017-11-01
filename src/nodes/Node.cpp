#include "Node.hpp"

Node::Node() : _subscribers() { }

Node::~Node() { }

void Node::AddSubscriber(Node::const_ptr const subscriber)
{
    _subscribers.insert(subscriber);
}

void Node::RemoveSubscriber(Node::const_ptr const subscriber)
{
    _subscribers.erase(subscriber);
}

Node::ConstNodeList Node::GetSubscribers() const
{
    return ConstNodeList(std::begin(_subscribers), std::end(_subscribers));
}
