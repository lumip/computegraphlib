#include <iostream>
#include <algorithm>
#include <random>

#include <papi.h>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "nodes/MatrixMultNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"
#include "ImplementationStrategyFactory.hpp"

int main(int argc, const char * const argv[])
{
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if(retval != PAPI_VER_CURRENT)
    {
        std::cout << "could not initialize PAPI" << std::endl;
        return -1;
    }

    size_t m = 1000;
    size_t n = 10000;
    size_t size = m * n;

    InputNode i1("x");
    InputNode i2("y");
    MatrixMultNode testMultNode(&i1, &i2);

    MemoryDimensions dims1 {m, n};
    MemoryDimensions dims2 {n, m};

    std::cout << "compiling graph and setting up runtime..." << std::endl;
    // provide input data dimensions
    InputDimensionsMap inputDimensions;
    inputDimensions.emplace("x", dims1);
    inputDimensions.emplace("y", dims2);

    // set up graph compilation context and platform
    ImplementationStrategyFactory fact;
    CompilationMemoryMap compilationMemoryMap(inputDimensions);
    std::unique_ptr<GraphCompilationPlatform> platform = fact.CreateGraphCompilationTargetStrategy(compilationMemoryMap);

    long long time_setup_start = PAPI_get_real_nsec();

    // set up working memory for input nodes (will usually be done during compilation if whole graph is compiled; testing only single node here)
    compilationMemoryMap.RegisterNodeMemory(&i1, dims1);
    compilationMemoryMap.RegisterNodeMemory(&i2, dims2);
    compilationMemoryMap.RegisterInputNode("x", &i1);
    compilationMemoryMap.RegisterInputNode("y", &i2);
    testMultNode.GetMemoryDimensions(compilationMemoryMap);

    platform->ReserveMemoryBuffer(&i1);
    platform->ReserveMemoryBuffer(&i2);
    platform->ReserveMemoryBuffer(&testMultNode);
    platform->AllocateAllMemory();

    // compile kernels
    i1.Compile(*platform);
    i2.Compile(*platform);
    testMultNode.Compile(*platform);

    long long time_setup_stop = PAPI_get_real_nsec();
    std::cout << "done setting up" << std::endl;

    // get mapped memory of working nodes and fill it with random data
    std::cout << "generating input data..." << std::endl;
    float* const mappedInput1 = platform->GetMappedInputBuffer("x");
    float* const mappedInput2 = platform->GetMappedInputBuffer("y");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-100.f, 100.f);

    std::generate(mappedInput1, mappedInput1 + size, [&dist, &gen]()->float {return dist(gen);});
    std::generate(mappedInput2, mappedInput2 + size, [&dist, &gen]()->float {return dist(gen);});
    std::cout << "done generating input data" << std::endl;

    std::cout << "copying data" << std::endl;
    long long time_copy_start = PAPI_get_real_nsec();

    // copy input data into node working memory (will usually be done by compiled kernels for InputNode if whole graph is run; testing only single node here)
    platform->CopyInputData(&i1, mappedInput1);
    platform->CopyInputData(&i2, mappedInput2);
    platform->WaitUntilDataTransferFinished();

    long long time_copy_stop = PAPI_get_real_nsec();
    std::cout << "running computation" << std::endl;
    long long time_start = PAPI_get_real_nsec();

    // run compiled kernel
    platform->Evaluate();
    platform->WaitUntilEvaluationFinished();

    long long time_stop = PAPI_get_real_nsec();

    std::cout << "Setup: " << time_setup_stop - time_setup_start << " ns; Copy: "<< time_copy_stop - time_copy_start << " ns; Compute: " << time_stop - time_start << " ns" << std::endl;
    return 0;
}
