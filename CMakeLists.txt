cmake_minimum_required (VERSION 3.1)

project(mpp_project)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CMAKE_CURRENT_SOURCE_DIR}/mnist")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Werror")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -Wall -Werror")

# Collect source files
file(GLOB_RECURSE Includes "includes/*hpp")
file(GLOB_RECURSE Sources "src/*.cpp")
file(GLOB_RECURSE CPUSources "src-cpu/*.cpp" "src-cpu/*.hpp")
file(GLOB_RECURSE GPUSources "src-gpu/*.cpp")
file(GLOB_RECURSE OCLKernels "src-opencl-kernels/*.c")
file(GLOB_RECURSE OCLKernelSources "src-opencl-kernels/*.cpp")
include_directories("includes")

#option(GPUImplementation "Compile with GPU accelerated OpenCL implementation of compute graph evaluation. Turn off for sequential CPU evaluation." OFF)

#if(GPUImplementation)
#    add_definitions(-DGPU)

# Handle OpenCL
find_package(OpenCL REQUIRED)
link_directories(${OpenCL_LIBRARY})

find_package(PAPI REQUIRED)

find_package(MNIST)

##########################################################################################################################
############################################### GRAPH COMPUTATION LIBRARY ################################################
##########################################################################################################################

add_library(cgLibCPU SHARED ${Includes} ${Sources} ${CPUSources})
target_compile_definitions(cgLibCPU PRIVATE -DCPU)

add_library(cgLibGPU SHARED ${Includes} ${Sources} ${GPUSources} ${OCLKernelSources})
target_compile_definitions(cgLibGPU PRIVATE -DGPU)
target_include_directories(cgLibGPU PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(cgLibGPU ${OpenCL_LIBRARY})

##########################################################################################################################
############ PLAYGROUND TESTBED: tests/main.cpp is a general playground for in-development testing #######################
##########################################################################################################################
add_executable(mainCPU "tests/main.cpp")
target_compile_definitions(mainCPU PRIVATE -DCPU)
target_link_libraries(mainCPU cgLibCPU)

add_executable(mainGPU "tests/main.cpp")
target_compile_definitions(mainGPU PRIVATE -DGPU)
target_include_directories(mainGPU PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(mainGPU ${OpenCL_LIBRARY} cgLibGPU)


##########################################################################################################################
################# SINGLE NODE TESTS: milestone 1, testing the implementation of single nodes #############################
##########################################################################################################################

# naming convention: 0sc01c is read as : 0s -> stage 0, single node test, c -> correctness test, 01 -> counter, c -> cpu implementation test
#                    0sc01g is read as : 0s -> stage 0, single node test, c -> correctness test, 01 -> counter, g -> gpu implementation test
#                    0sp01c is read as : 0s -> stage 0, single node test, p -> performance test, 01 -> counter, c -> cpu implementation test
# overall format: <milestone number><s(ingle)|g(raph)><c(orrectness)|p(erformance)><test number><c(pu)|g(pu)>

###### Correctness tests on CPU
add_executable(0sc01c_VectorAddNodeSingle "tests/0sc01-VectorAddNodeSingle.cpp")
target_compile_definitions(0sc01c_VectorAddNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc01c_VectorAddNodeSingle cgLibCPU)
add_executable(0sc02c_InputNodeSingle "tests/0sc02-InputNodeSingle.cpp")
target_compile_definitions(0sc02c_InputNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc02c_InputNodeSingle cgLibCPU)
add_executable(0sc03c_MatrixMultNodeSingle "tests/0sc03-MatrixMultNodeSingle.cpp")
target_compile_definitions(0sc03c_MatrixMultNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc03c_MatrixMultNodeSingle cgLibCPU)
add_executable(0sc04c_VectorDivNodeSingle "tests/0sc04-VectorDivNodeSingle.cpp")
target_compile_definitions(0sc04c_VectorDivNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc04c_VectorDivNodeSingle cgLibCPU)
add_executable(0sc05c_VectorMultNodeSingle "tests/0sc05-VectorMultNodeSingle.cpp")
target_compile_definitions(0sc05c_VectorMultNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc05c_VectorMultNodeSingle cgLibCPU)
add_executable(0sc06c_ReduceMeanNodeSingle "tests/0sc06-ReduceMeanNodeSingle.cpp")
target_compile_definitions(0sc06c_ReduceMeanNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc06c_ReduceMeanNodeSingle cgLibCPU)
add_executable(0sc07c_SliceNodeSingle "tests/0sc07-SliceNodeSingle.cpp")
target_compile_definitions(0sc07c_SliceNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc07c_SliceNodeSingle cgLibCPU)
add_executable(0sc08c_StackNodeSingle "tests/0sc08-StackNodeSingle.cpp")
target_compile_definitions(0sc08c_StackNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc08c_StackNodeSingle cgLibCPU)
add_executable(0sc09c_TranposeNodeSingle "tests/0sc09-TransposeNodeSingle.cpp")
target_compile_definitions(0sc09c_TranposeNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc09c_TranposeNodeSingle cgLibCPU)
add_executable(0sc10c_ExpFuncNodeSingle "tests/0sc10-ExpFuncNodeSingle.cpp")
target_compile_definitions(0sc10c_ExpFuncNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc10c_ExpFuncNodeSingle cgLibCPU)

###### Performance tests on CPU
add_executable(0sp01c_VectorAddNodeSingle "tests/0sp01-VectorAddNodeSingle.cpp")
target_compile_definitions(0sp01c_VectorAddNodeSingle PRIVATE -DCPU)
target_include_directories(0sp01c_VectorAddNodeSingle PRIVATE ${PAPI_INCLUDE_DIRS})
target_link_libraries(0sp01c_VectorAddNodeSingle cgLibCPU ${PAPI_LIBRARIES})
add_executable(0sp03c_MatrixMultNodeSingle "tests/0sp03-MatrixMultNodeSingle.cpp")
target_compile_definitions(0sp03c_MatrixMultNodeSingle PRIVATE -DCPU)
target_include_directories(0sp03c_MatrixMultNodeSingle PRIVATE ${PAPI_INCLUDE_DIRS})
target_link_libraries(0sp03c_MatrixMultNodeSingle cgLibCPU ${PAPI_LIBRARIES})

###### Correctness tests on GPU
add_executable(0sc01g_VectorAddNodeSingle "tests/0sc01-VectorAddNodeSingle.cpp")
target_compile_definitions(0sc01g_VectorAddNodeSingle PRIVATE -DGPU)
target_include_directories(0sc01g_VectorAddNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(0sc01g_VectorAddNodeSingle ${OpenCL_LIBRARY} cgLibGPU ${PAPI_LIBRARIES})
add_executable(0sc02g_InputNodeSingle "tests/0sc02-InputNodeSingle.cpp")
target_compile_definitions(0sc02g_InputNodeSingle PRIVATE -DGPU)
target_include_directories(0sc02g_InputNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(0sc02g_InputNodeSingle ${OpenCL_LIBRARY} cgLibGPU)
add_executable(0sc03g_MatrixMultNodeSingle "tests/0sc03-MatrixMultNodeSingle.cpp")
target_compile_definitions(0sc03g_MatrixMultNodeSingle PRIVATE -DGPU)
target_include_directories(0sc03g_MatrixMultNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(0sc03g_MatrixMultNodeSingle ${OpenCL_LIBRARY} cgLibGPU)
add_executable(0sc04g_VectorDivNodeSingle "tests/0sc04-VectorDivNodeSingle.cpp")
target_compile_definitions(0sc04g_VectorDivNodeSingle PRIVATE -DGPU)
target_include_directories(0sc04g_VectorDivNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(0sc04g_VectorDivNodeSingle ${OpenCL_LIBRARY} cgLibGPU)
add_executable(0sc05g_VectorMultNodeSingle "tests/0sc05-VectorMultNodeSingle.cpp")
target_compile_definitions(0sc05g_VectorMultNodeSingle PRIVATE -DGPU)
target_include_directories(0sc05g_VectorMultNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(0sc05g_VectorMultNodeSingle ${OpenCL_LIBRARY} cgLibGPU)
add_executable(0sc06g_ReduceMeanNodeSingle "tests/0sc06-ReduceMeanNodeSingle.cpp")
target_compile_definitions(0sc06g_ReduceMeanNodeSingle PRIVATE -DGPU)
target_include_directories(0sc06g_ReduceMeanNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(0sc06g_ReduceMeanNodeSingle ${OpenCL_LIBRARY} cgLibGPU)
add_executable(0sc07g_SliceNodeSingle "tests/0sc07-SliceNodeSingle.cpp")
target_compile_definitions(0sc07g_SliceNodeSingle PRIVATE -DGPU)
target_include_directories(0sc07g_SliceNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(0sc07g_SliceNodeSingle ${OpenCL_LIBRARY} cgLibGPU)
add_executable(0sc08g_StackNodeSingle "tests/0sc08-StackNodeSingle.cpp")
target_compile_definitions(0sc08g_StackNodeSingle PRIVATE -DGPU)
target_include_directories(0sc08g_StackNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(0sc08g_StackNodeSingle ${OpenCL_LIBRARY} cgLibGPU)
add_executable(0sc09g_TranposeNodeSingle "tests/0sc09-TransposeNodeSingle.cpp")
target_compile_definitions(0sc09g_TranposeNodeSingle PRIVATE -DGPU)
target_include_directories(0sc09g_TranposeNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(0sc09g_TranposeNodeSingle ${OpenCL_LIBRARY} cgLibGPU)
add_executable(0sc10g_ExpFuncNodeSingle "tests/0sc10-ExpFuncNodeSingle.cpp")
target_compile_definitions(0sc10g_ExpFuncNodeSingle PRIVATE -DGPU)
target_include_directories(0sc10g_ExpFuncNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(0sc10g_ExpFuncNodeSingle ${OpenCL_LIBRARY} cgLibGPU)

###### Performance tests on GPU
add_executable(0sp01g_VectorAddNodeSingle "tests/0sp01-VectorAddNodeSingle.cpp")
target_compile_definitions(0sp01g_VectorAddNodeSingle PRIVATE -DGPU)
target_include_directories(0sp01g_VectorAddNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS} ${PAPI_INCLUDE_DIRS})
target_link_libraries(0sp01g_VectorAddNodeSingle ${OpenCL_LIBRARY} cgLibGPU ${PAPI_LIBRARIES})
add_executable(0sp03g_MatrixMultNodeSingle "tests/0sp03-MatrixMultNodeSingle.cpp")
target_compile_definitions(0sp03g_MatrixMultNodeSingle PRIVATE -DGPU)
target_include_directories(0sp03g_MatrixMultNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS} ${PAPI_INCLUDE_DIRS})
target_link_libraries(0sp03g_MatrixMultNodeSingle ${OpenCL_LIBRARY} cgLibGPU ${PAPI_LIBRARIES})

##########################################################################################################################
######################## GRAPH TESTS: milestone 2, testing the implementation of a full graph ############################
##########################################################################################################################
###### Correctness tests on CPU
add_executable(1gc01c_Graph "tests/1gc01-Graph.cpp")
target_compile_definitions(1gc01c_Graph PRIVATE -DCPU)
target_include_directories(1gc01c_Graph PRIVATE ${MNIST_INCLUDE_DIR})
target_link_libraries(1gc01c_Graph cgLibCPU)

###### Performance tests on CPU
add_executable(1gp02c_MNISTGraphForwardOnly "tests/1gp02-MNISTGraphForwardOnly.cpp")
target_compile_definitions(1gp02c_MNISTGraphForwardOnly PRIVATE -DCPU)
target_include_directories(1gp02c_MNISTGraphForwardOnly PRIVATE ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
target_link_libraries(1gp02c_MNISTGraphForwardOnly cgLibCPU ${PAPI_LIBRARIES})
add_executable(1gp03c_MNISTGraph "tests/1gp03-MNISTGraph.cpp")
target_compile_definitions(1gp03c_MNISTGraph PRIVATE -DCPU)
target_include_directories(1gp03c_MNISTGraph PRIVATE ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
target_link_libraries(1gp03c_MNISTGraph cgLibCPU ${PAPI_LIBRARIES})

###### Correctness tests on GPU
add_executable(1gc01g_Graph "tests/1gc01-Graph.cpp")
target_compile_definitions(1gc01g_Graph PRIVATE -DGPU)
target_include_directories(1gc01g_Graph PRIVATE ${OpenCL_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
target_link_libraries(1gc01g_Graph ${OpenCL_LIBRARY} cgLibGPU)

###### Performance tests on GPU
add_executable(1gp02g_MNISTGraphFordwardOnly "tests/1gp02-MNISTGraphForwardOnly.cpp")
target_compile_definitions(1gp02g_MNISTGraphFordwardOnly PRIVATE -DGPU)
target_include_directories(1gp02g_MNISTGraphFordwardOnly PRIVATE ${OpenCL_INCLUDE_DIRS} ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
target_link_libraries(1gp02g_MNISTGraphFordwardOnly ${OpenCL_LIBRARY} cgLibGPU ${PAPI_LIBRARIES})
add_executable(1gp03g_MNISTGraph "tests/1gp03-MNISTGraph.cpp")
target_compile_definitions(1gp03g_MNISTGraph PRIVATE -DGPU)
target_include_directories(1gp03g_MNISTGraph PRIVATE ${OpenCL_INCLUDE_DIRS} ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
target_link_libraries(1gp03g_MNISTGraph ${OpenCL_LIBRARY} cgLibGPU ${PAPI_LIBRARIES})

###### MNIST tasks baseline/benchmark non-abstracted CPU implementation
add_executable(1gp02b_MNISTGraphForwardOnlyBaseline "tests/1gp02b-MNISTGraphForwardOnlyBaseline.cpp")
target_compile_definitions(1gp02b_MNISTGraphForwardOnlyBaseline PRIVATE)
target_include_directories(1gp02b_MNISTGraphForwardOnlyBaseline PRIVATE ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
target_link_libraries(1gp02b_MNISTGraphForwardOnlyBaseline ${PAPI_LIBRARIES})
add_executable(1gp03b_MNISTGraphBaseline "tests/1gp03b-MNISTGraphBaseline.cpp")
target_compile_definitions(1gp03b_MNISTGraphBaseline PRIVATE)
target_include_directories(1gp03b_MNISTGraphBaseline PRIVATE ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
target_link_libraries(1gp03b_MNISTGraphBaseline ${PAPI_LIBRARIES})
