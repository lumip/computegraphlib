#include "GraphCompilationCPUPlatform.hpp"

#include "CompilationMemoryMap.hpp"

#include "nodes/InputNode.hpp"
#include "nodes/MatrixMultNode.hpp"
#include "nodes/VariableNode.hpp"
#include "nodes/VectorAddNode.hpp"

void GraphCompilationCPUPlatform::CompileInputNode(const InputNode* const node) { }

class MatrixMultNodeCPUKernel : public Kernel
{
private:
    const float* const _memA;
    const float* const _memB;
    float* const _memRes;
    const size_t _m;
    const size_t _n;
    const size_t _d;
private:
    inline size_t GetIndex(size_t i, size_t j, size_t stride) const
    {
        return i * stride + j;
    }
public:
    MatrixMultNodeCPUKernel(const float* const memA, const float* const memB, float* const memRes, const size_t m, const size_t n, const size_t d)
        : _memA(memA), _memB(memB), _memRes(memRes), _m(m), _n(n), _d(d)
    { }

    virtual ~MatrixMultNodeCPUKernel() { }

    void Run()
    {
        /*for (size_t i = 0; i < _m; ++i)
        {
            for (size_t j = 0; j < _n; ++j)
            {
                float r_ij = 0.0f;
                for (size_t k = 0; k < _d; ++k)
                {
                    r_ij += _memA[GetIndex(i, k, _d)] * _memB[GetIndex(k, j, _n)];
                }
                _memRes[GetIndex(i, j, _n)] = r_ij;
            }
        }*/
        // the above is probably the most inefficient thing (that poor cache :/ )
        // the below is faster by a factor of ~8 (tested)
        std::fill_n(_memRes, _m * _n, 0.0f);
        for (size_t i = 0; i < _m; ++i)
        {
            for (size_t k = 0; k < _d; ++k)
            {
                float a_ik = _memA[GetIndex(i, k, _d)]; // linear access
                for (size_t j = 0; j < _n; ++j)
                {
                    // GetIndex(i, j, _n) = i * _n + j -> size_t base = i * _n; and ++base in every iteration; however, that's not faster (tested) and like it is now it's more readable
                    _memRes[GetIndex(i, j, _n)] += a_ik * _memB[GetIndex(k, j, _n)]; // linear access to both
                }
            }
        }
    }
};

void GraphCompilationCPUPlatform::CompileMatrixMultNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const MatrixMultNode* const node)
{
    MemoryDimensions inputADims = _dimensionsMap.GetNodeMemoryDimensions(inputANode);
    MemoryDimensions inputBDims = _dimensionsMap.GetNodeMemoryDimensions(inputBNode);
    const MemoryHandle inputABuffer = _bufferMap.at(inputANode).get();
    const MemoryHandle inputBBuffer = _bufferMap.at(inputBNode).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    auto m = inputADims.yDim;
    auto n = inputBDims.xDim;
    auto d = inputADims.xDim;
    _kernels.emplace_back(
           std::unique_ptr<Kernel>(new MatrixMultNodeCPUKernel(inputABuffer,
                                                               inputBBuffer,
                                                               resultBuffer,
                                                               m, n, d))
    );
}

void GraphCompilationCPUPlatform::CompileVariableNode(const VariableNode* const node) { }

class VectorAddNodeCPUKernel : public Kernel
{
private:
    const float* const _memoryA;
    const float* const _memoryB;
    float* const _memoryResult;
    const size_t _size;
public:
    VectorAddNodeCPUKernel(const float* const memoryA, const float* const memoryB, float* const memoryResult, size_t size)
        : _memoryA(memoryA), _memoryB(memoryB), _memoryResult(memoryResult), _size(size)
    { }

    virtual ~VectorAddNodeCPUKernel() { }

    void Run()
    {
        for (size_t i = 0; i < _size; ++i)
        {
            _memoryResult[i] = _memoryA[i] + _memoryB[i];
        }
    }
};

void GraphCompilationCPUPlatform::CompileVectorAddNode(const ConstNodePtr inputANode, const ConstNodePtr inputBNode, const VectorAddNode* const node)
{
    MemoryDimensions resultDims = _dimensionsMap.GetNodeMemoryDimensions(node);
    const MemoryHandle inputABuffer = _bufferMap.at(inputANode).get();
    const MemoryHandle inputBBuffer = _bufferMap.at(inputBNode).get();
    const MemoryHandle resultBuffer = _bufferMap.at(node).get();
    _kernels.emplace_back(
           std::unique_ptr<Kernel>(new VectorAddNodeCPUKernel(inputABuffer,
                                                              inputBBuffer,
                                                              resultBuffer,
                                                              resultDims.size()))
    );
}
