#include "NodeCompilerCPU.hpp"

NodeCompilerCPU::NodeCompilerCPU() { }

NodeCompilerCPU::~NodeCompilerCPU() { }

std::unique_ptr<Kernel> NodeCompilerCPU::CompileInputNode(const InputNode * const node) { }

class MatrixMultNodeCPUKernel : public Kernel
{
private:
    const float* const _memoryA;
    const float* const _memoryB;
    float* const _memoryResult;
    const size_t _m;
    const size_t _n;
    const size_t _d;
private:
    inline size_t GetIndex(size_t i, size_t j, size_t stride) const // row-major; todo: does it really inline?
    {
        return i * stride + j;
    }
public:
    MatrixMultNodeCPUKernel(const float* const memoryA, const float* const memoryB, const NodeMemoryDescriptor memoryResDesc, const size_t vectorDim)
        : _memoryA(memoryA)
        , _memoryB(memoryB)
        , _memoryResult(reinterpret_cast<float* const>(memoryResDesc.handle))
        , _m(memoryResDesc.dimensions.yDim)
        , _n(memoryResDesc.dimensions.xDim)
        , _d(vectorDim)
    { }

    virtual ~MatrixMultNodeCPUKernel() { }

    void Run()
    {
        for (size_t i = 0; i < _m; ++i)
        {
            for (size_t j = 0; j < _n; ++j)
            {
                float r_ij = 0.0f;
                for (size_t k = 0; k < _d; ++k)
                {
                    r_ij += _memoryA[GetIndex(i, k, _d)] * _memoryB[GetIndex(k, j, _n)];
                }
                _memoryResult[GetIndex(i, j, _n)] = r_ij;
            }
        }
        // todo: the above is probably the most inefficient thing (that poor cache :/ )
        // the below is probably better: benchmark
        /*
        std::fill_n(_memoryResult, _m * _n, 0.0f);
        for (size_t i = 0; i < _m; ++i)
        {
            for (size_t k = 0; k < _d; ++k)
            {
                float a_ik = _memoryA[GetIndex(i, k, _d)]; // linear access
                for (size_t j = 0; j < _n; ++j)
                {
                    // todo: GetIndex(i, j, _n) = i * _n + j -> size_t base = i * _n; and ++base in every iteration; is that faster?
                    _memoryResult[GetIndex(i, j, _n)] += a_ik * _memoryB[GetIndex(k, j, _n)]; // linear access to both
                }
            }
        }*/
    }
};

std::unique_ptr<Kernel> NodeCompilerCPU::CompileMatrixMultNode(const NodeMemoryDescriptor inputAMem, const NodeMemoryDescriptor inputBMem, const NodeMemoryDescriptor resultMem)
{
    return std::unique_ptr<Kernel>(new MatrixMultNodeCPUKernel(reinterpret_cast<const float* const>(inputAMem.handle),
                                                               reinterpret_cast<const float* const>(inputBMem.handle),
                                                               resultMem,
                                                               inputAMem.dimensions.xDim));
}

std::unique_ptr<Kernel> NodeCompilerCPU::CompileVariableNode(const VariableNode * const node) { }

class VectorAddNodeCPUKernel : public Kernel
{
private:
    const float* const _memoryA;
    const float* const _memoryB;
    float* const _memoryResult;
    const size_t _size;
public:
    VectorAddNodeCPUKernel(const float* const memoryA, const float* const memoryB, float* const memoryResult, size_t size)
        : _memoryA(memoryA)
        , _memoryB(memoryB)
        , _memoryResult(memoryResult)
        , _size(size)
    {
    }
    virtual ~VectorAddNodeCPUKernel() { }
    void Run()
    {
        for (size_t i = 0; i < _size; ++i)
        {
            _memoryResult[i] = _memoryA[i] + _memoryB[i];
        }
    }
};

std::unique_ptr<Kernel> NodeCompilerCPU::CompileVectorAddNode(const NodeMemoryDescriptor inputAMem, const NodeMemoryDescriptor inputBMem, const NodeMemoryDescriptor resultMem)
{
    return std::unique_ptr<Kernel>(new VectorAddNodeCPUKernel(reinterpret_cast<const float* const>(inputAMem.handle),
                                                              reinterpret_cast<const float* const>(inputBMem.handle),
                                                              reinterpret_cast<float* const>(resultMem.handle),
                                                              resultMem.dimensions.size()));
}
