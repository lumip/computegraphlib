#ifdef CPU
#include "nodes/MatrixMultNode.hpp"
#include "GraphCompilationContext.hpp"
#include "Kernel.hpp"

class MatrixMultCPUKernel : public Kernel
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
    MatrixMultCPUKernel(const float* const memoryA, const float* const memoryB, const NodeMemoryDescriptor memoryResDesc, const size_t vectorDim)
        : _memoryA(memoryA)
        , _memoryB(memoryB)
        , _memoryResult(reinterpret_cast<float* const>(memoryResDesc.handle))
        , _m(memoryResDesc.dimensions.yDim)
        , _n(memoryResDesc.dimensions.xDim)
        , _d(vectorDim)
    { }

    virtual ~MatrixMultCPUKernel() { }

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

void MatrixMultNode::Compile(GraphCompilationContext& context) const
{
    NodeMemoryDescriptor memDescA = context.GetNodeMemoryDescriptor(this->_a);
    NodeMemoryDescriptor memDescB = context.GetNodeMemoryDescriptor(this->_b);
    if (memDescA.dimensions.xDim != memDescB.dimensions.yDim)
    {
        throw new std::invalid_argument("Matrix dimensions do not agree.");
    }
    auto m = memDescA.dimensions.yDim;
    auto d = memDescA.dimensions.xDim; // == memDescB.yDim;
    auto n = memDescB.dimensions.xDim;
    const NodeMemoryDescriptor memDesc = context.RegisterMemory({m, n});
    context.AssignNodeMemory(this, memDesc.handle);
    const float* const inputAMemBuffer = reinterpret_cast<const float* const>(memDescA.handle);
    const float* const inputBMemBuffer = reinterpret_cast<const float* const>(memDescB.handle);
    context.EnqueueKernel(std::unique_ptr<Kernel>(new MatrixMultCPUKernel(inputAMemBuffer, inputBMemBuffer, memDesc, d))); // std::make_unique only since c++14
}

#endif
