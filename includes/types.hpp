#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <map>

//typedef std::vector<float> DataBuffer;
//typedef const DataBuffer ConstDataBuffer;
//typedef ConstDataBuffer InputDataBuffer;
//typedef InputDataBuffer ConstDataBuffer;
typedef float* DataBufferHandle;
typedef const float* ConstDataBufferHandle;
typedef ConstDataBufferHandle InputDataBufferHandle;
typedef std::map<std::string, InputDataBufferHandle> InputDataMap;

#endif
