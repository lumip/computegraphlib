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
    union {
        struct {
            size_t yDim;
            size_t xDim;
        };
        size_t dims[2];
    };
    size_t size() const
    {
        return xDim * yDim;
    }
};
inline bool operator ==(const MemoryDimensions& lhs, const MemoryDimensions& rhs)
{
    return lhs.yDim == rhs.yDim && lhs.xDim == rhs.xDim;
}
inline bool operator !=(const MemoryDimensions& lhs, const MemoryDimensions& rhs)
{
    return !(lhs == rhs);
}

typedef std::map<std::string, const MemoryDimensions> InputDimensionsMap;

class Node;
typedef Node* NodePtr;
typedef const Node* ConstNodePtr;

#endif
