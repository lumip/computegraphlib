#include "MatrixMultNodeCPUKernel.hpp"

#include <assert.h>

MatrixMultNodeCPUKernel::MatrixMultNodeCPUKernel(const float* const memA, const float* const memB, float* const memRes, const size_t m, const size_t n, const size_t d)
    : _memA(memA), _memB(memB), _memRes(memRes), _m(m), _n(n), _d(d)
{
    assert(_memA != nullptr);
    assert(_memB != nullptr);
    assert(_memRes != nullptr);
}

MatrixMultNodeCPUKernel::~MatrixMultNodeCPUKernel() { }

void MatrixMultNodeCPUKernel::Run()
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
