#include <iostream>
#include <algorithm>
#include <random>
#include <sstream>

#include <papi.h>

#include "types.hpp"
#include "nodes/InputNode.hpp"
#include "nodes/StackNode.hpp"
#include "CompilationMemoryMap.hpp"
#include "GraphCompilationPlatform.hpp"
#include "GraphCompilationPlatformFactory.hpp"

int main(int argc, const char * const argv[])
{
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if(retval != PAPI_VER_CURRENT)
    {
        std::cout << "could not initialize PAPI" << std::endl;
        return -1;
    }

    size_t size = 12000;
    size_t axis = 1; // TWEAK HERE

    // set up slice dimensions according to tested axis
    MemoryDimensions sliceDim { 1, size }; // if stacking along y dimension, slices are rows
    if (axis == 1) // if stacking along x dimension, slices are columns (swap dimensions)
    {
        std::swap(sliceDim.yDim, sliceDim.xDim);
    }

    std::cout << "compiling graph and setting up runtime..." << std::endl;
    // Create, and set up InputNodes holding the slices data
    GraphCompilationPlatformFactory fact;
    CompilationMemoryMap compilationMemoryMap;
    std::vector<std::unique_ptr<InputNode>> createdNodes(size); // keep track of dynamically allocated nodes so that they get freed upon returning
    std::vector<NodePtr> sliceInputs(size); // vector holding plain pointers to feed into StackNode
    std::unique_ptr<GraphCompilationPlatform> platform = fact.CreateGraphCompilationTargetStrategy(compilationMemoryMap);

    long long time_setup_start = PAPI_get_real_nsec();
    for (size_t i = 0; i < size; ++i)
    {
        std::stringstream ss;
        ss << i;
        std::string const nodeName = ss.str();
        InputNode* const inputNode = new InputNode(nodeName);
        createdNodes[i] = std::unique_ptr<InputNode>(inputNode);
        sliceInputs[i] = inputNode;
        compilationMemoryMap.RegisterNodeMemory(inputNode, sliceDim);
        compilationMemoryMap.RegisterInputNode(nodeName, inputNode);
        platform->ReserveMemoryBuffer(inputNode);
        inputNode->Compile(*platform);
    }

    // create, set up and compile StackNode
    StackNode node(sliceInputs, axis);
    node.GetMemoryDimensions(compilationMemoryMap);
    platform->ReserveMemoryBuffer(&node);
    platform->AllocateAllMemory();

    // compile kernels
    node.Compile(*platform);

    long long time_setup_stop = PAPI_get_real_nsec();
    std::cout << "done setting up" << std::endl;

    // get mapped memory of working nodes and fill it with random data
    std::cout << "generating input data..." << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-100.f, 100.f);

    for (std::string const& inputName : compilationMemoryMap.GetInputNames())
    {
        float* const mappedBuffer = platform->GetMappedInputBuffer(inputName);
        std::generate_n(mappedBuffer, size, [&dist, &gen]()->float {return dist(gen);});
    }
    std::cout << "done generating input data" << std::endl;

    std::cout << "copying data" << std::endl;
    long long time_copy_start = PAPI_get_real_nsec();

    // copy input data into node working memory (will usually be done by compiled kernels for InputNode if whole graph is run; testing only single node here)
    for (std::unique_ptr<InputNode>& inputNode : createdNodes)
    {
        platform->CopyInputData(inputNode.get(), platform->GetMappedInputBuffer(inputNode->GetName()));
    }
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
