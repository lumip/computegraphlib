#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <map>
#include <vector>
#include <functional>

typedef std::vector<float> DataBuffer;
typedef const DataBuffer ConstDataBuffer;
typedef ConstDataBuffer InputDataBuffer;

typedef std::map<std::string, std::reference_wrapper<InputDataBuffer>> InputDataMap;

class Node;
typedef Node* NodePtr;
typedef const Node* ConstNodePtr;

#endif
