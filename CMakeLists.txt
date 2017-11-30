cmake_minimum_required (VERSION 3.1)

project(mpp_project)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CMAKE_CURRENT_SOURCE_DIR}/mnist")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
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
########### GRAPH TESTS: milestone 2, testing the implementation of a full graph using the MNIST classifier ##############
##########################################################################################################################
add_executable(1gc01c_MNISTGraph "tests/1gc01-MNISTGraph.cpp")
target_compile_definitions(1gc01c_MNISTGraph PRIVATE -DCPU -DMNIST_DATA_DIR="${MNIST_DATA_DIR}") #todo: do not use a define to pass the data location
target_include_directories(1gc01c_MNISTGraph PRIVATE ${MNIST_INCLUDE_DIR})
target_link_libraries(1gc01c_MNISTGraph cgLibCPU)

add_executable(1gc01g_MNISTGraph "tests/1gc01-MNISTGraph.cpp")
target_compile_definitions(1gc01g_MNISTGraph PRIVATE -DGPU -DMNIST_DATA_DIR="${MNIST_DATA_DIR}")
target_include_directories(1gc01g_MNISTGraph PRIVATE ${OpenCL_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
target_link_libraries(1gc01g_MNISTGraph ${OpenCL_LIBRARY} cgLibGPU)
