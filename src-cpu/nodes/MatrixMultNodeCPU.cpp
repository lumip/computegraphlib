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
    MatrixMultCPUKernel(const float* const memoryA, const float* const memoryB, const GraphCompilationContext::NodeMemoryDescriptor memoryResDesc, const size_t vectorDim)
        : _memoryA(memoryA)
        , _memoryB(memoryB)
        , _memoryResult(memoryResDesc.handle)
        , _m(memoryResDesc.n)
        , _n(memoryResDesc.dimensions)
        , _d(vectorDim)
    { }

    virtual ~MatrixMultCPUKernel() { }

    void Run() const
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

std::unique_ptr<const Kernel> MatrixMultNode::Compile(GraphCompilationContext* const context) const
{
    GraphCompilationContext::NodeMemoryDescriptor memDescA = context->GetNodeMemoryDescriptor(this->_a);
    GraphCompilationContext::NodeMemoryDescriptor memDescB = context->GetNodeMemoryDescriptor(this->_b);
    if (memDescA.dimensions != memDescB.n)
    {
        throw new std::invalid_argument("Matrix dimensions do not agree.");
    }
    GraphCompilationContext::NodeMemoryHandle mem = context->RegisterMemory(memDescA.n, memDescB.dimensions);
    context->AssignNodeMemory(this, mem);
    GraphCompilationContext::NodeMemoryDescriptor desc = context->GetNodeMemoryDescriptor(this);
    return std::unique_ptr<const Kernel>(new MatrixMultCPUKernel(memDescA.handle, memDescB.handle, desc, memDescA.dimensions)); // std::make_unique only since c++14
}

#endif
