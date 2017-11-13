#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <map>
#include <vector>
#include <functional>

typedef std::vector<float> DataBuffer; // todo: consider using std::valarray
typedef const DataBuffer ConstDataBuffer;
typedef ConstDataBuffer InputDataBuffer;

typedef std::map<std::string, std::reference_wrapper<InputDataBuffer>> InputDataMap;
struct MemoryDimensions
{
    size_t yDim;
    size_t xDim;
    size_t size() const
    {
        return xDim * yDim;
    }
};
typedef std::map<std::string, const MemoryDimensions> InputDimensionsMap;

class Node;
typedef Node* NodePtr;
typedef const Node* ConstNodePtr;

#endif
