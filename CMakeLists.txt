cmake_minimum_required (VERSION 3.1)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(mpp_project)

set (CMAKE_CXX_STANDARD 11)

# Collect source files
file(GLOB_RECURSE Includes "includes/*hpp")
file(GLOB_RECURSE Sources "src/*.cpp")
file(GLOB_RECURSE CPUSources "src-cpu/*.cpp")
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

add_library(cgLibCPU SHARED ${Includes} ${Sources} ${CPUSources})
target_compile_definitions(cgLibCPU PRIVATE -DCPU)

add_library(cgLibGPU SHARED ${Includes} ${Sources} ${GPUSources} ${OCLKernelSources})
target_compile_definitions(cgLibGPU PRIVATE -DGPU)
target_include_directories(cgLibGPU PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(cgLibGPU ${OpenCL_LIBRARY})

add_executable(mainCPU "tests/main.cpp")
target_compile_definitions(mainCPU PRIVATE -DCPU)
target_link_libraries(mainCPU cgLibCPU)

add_executable(mainGPU "tests/main.cpp")
target_compile_definitions(mainGPU PRIVATE -DGPU)
target_include_directories(mainGPU PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(mainGPU ${OpenCL_LIBRARY} cgLibGPU)

# naming convention: 0sc01c is read as : 0s -> stage 0, single node test, c -> correctness test, 01 -> counter, c -> cpu implementation test
#                    0sc01g is read as : 0s -> stage 0, single node test, c -> correctness test, 01 -> counter, g -> gpu implementation test
#                    0sp01c is read as : 0s -> stage 0, single node test, p -> performance test, 01 -> counter, c -> cpu implementation test
add_executable(0sc01c_VectorAddNodeSingleCPU "tests/0sc01-VectorAddNodeSingle.cpp")
target_compile_definitions(0sc01c_VectorAddNodeSingleCPU PRIVATE -DCPU)
target_link_libraries(0sc01c_VectorAddNodeSingleCPU cgLibCPU)
add_executable(0sc02c_InputNodeSingleCPU "tests/0sc02-InputNodeSingle.cpp")
target_compile_definitions(0sc02c_InputNodeSingleCPU PRIVATE -DCPU)
target_link_libraries(0sc02c_InputNodeSingleCPU cgLibCPU)
add_executable(0sc03c_MatrixMultNodeSingleCPU "tests/0sc03-MatrixMultNodeSingle.cpp")
target_compile_definitions(0sc03c_MatrixMultNodeSingleCPU PRIVATE -DCPU)
target_link_libraries(0sc03c_MatrixMultNodeSingleCPU cgLibCPU)

add_executable(0sp01c_VectorAddNodeSingleCPU "tests/0sp01-VectorAddNodeSingle.cpp")
target_compile_definitions(0sp01c_VectorAddNodeSingleCPU PRIVATE -DCPU)
target_include_directories(0sp01c_VectorAddNodeSingleCPU PRIVATE ${PAPI_INCLUDE_DIRS})
target_link_libraries(0sp01c_VectorAddNodeSingleCPU cgLibCPU ${PAPI_LIBRARIES})
add_executable(0sp03c_MatrixMultNodeSingleCPU "tests/0sp03-MatrixMultNodeSingle.cpp")
target_compile_definitions(0sp03c_MatrixMultNodeSingleCPU PRIVATE -DCPU)
target_include_directories(0sp03c_MatrixMultNodeSingleCPU PRIVATE ${PAPI_INCLUDE_DIRS})
target_link_libraries(0sp03c_MatrixMultNodeSingleCPU cgLibCPU ${PAPI_LIBRARIES})

add_executable(0sc01g_VectorAddNodeSingleGPU "tests/0sc01-VectorAddNodeSingle.cpp")
target_compile_definitions(0sc01g_VectorAddNodeSingleGPU PRIVATE -DGPU)
target_include_directories(0sc01g_VectorAddNodeSingleGPU PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(0sc01g_VectorAddNodeSingleGPU ${OpenCL_LIBRARY} cgLibGPU ${PAPI_LIBRARIES})
add_executable(0sc02g_InputNodeSingleGPU "tests/0sc02-InputNodeSingle.cpp")
target_compile_definitions(0sc02g_InputNodeSingleGPU PRIVATE -DGPU)
target_include_directories(0sc02g_InputNodeSingleGPU PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(0sc02g_InputNodeSingleGPU ${OpenCL_LIBRARY} cgLibGPU)
add_executable(0sc03c_MatrixMultNodeSingleGPU "tests/0sc03-MatrixMultNodeSingle.cpp")
target_compile_definitions(0sc03c_MatrixMultNodeSingleGPU PRIVATE -DGPU)
target_include_directories(0sc03c_MatrixMultNodeSingleGPU PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(0sc03c_MatrixMultNodeSingleGPU ${OpenCL_LIBRARY} cgLibGPU)

add_executable(0sp01g_VectorAddNodeSingleGPU "tests/0sp01-VectorAddNodeSingle.cpp")
target_compile_definitions(0sp01g_VectorAddNodeSingleGPU PRIVATE -DGPU)
target_include_directories(0sp01g_VectorAddNodeSingleGPU PRIVATE ${OpenCL_INCLUDE_DIRS} ${PAPI_INCLUDE_DIRS})
target_link_libraries(0sp01g_VectorAddNodeSingleGPU ${OpenCL_LIBRARY} cgLibGPU ${PAPI_LIBRARIES})
add_executable(0sp03g_MatrixMultNodeSingleGPU "tests/0sp03-MatrixMultNodeSingle.cpp")
target_compile_definitions(0sp03g_MatrixMultNodeSingleGPU PRIVATE -DGPU)
target_include_directories(0sp03g_MatrixMultNodeSingleGPU PRIVATE ${OpenCL_INCLUDE_DIRS} ${PAPI_INCLUDE_DIRS})
target_link_libraries(0sp03g_MatrixMultNodeSingleGPU ${OpenCL_LIBRARY} cgLibGPU ${PAPI_LIBRARIES})
