cmake_minimum_required (VERSION 3.1)

project(mpp_project)

set (CMAKE_CXX_STANDARD 11)

# Collect source files
file(GLOB_RECURSE Includes "includes/*hpp")
file(GLOB_RECURSE Sources "src/*.cpp")
file(GLOB_RECURSE CPUSources "src-cpu/*.cpp")
file(GLOB_RECURSE GPUSources "src-gpu/*.cpp")
include_directories("includes")

#option(GPUImplementation "Compile with GPU accelerated OpenCL implementation of compute graph evaluation. Turn off for sequential CPU evaluation." OFF)

#if(GPUImplementation)
#    add_definitions(-DGPU)

# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(mainCPU "tests/main.cpp" ${Includes} ${Sources} ${CPUSources} ${GPUSources})
target_compile_definitions(mainCPU PRIVATE -DCPU)

add_executable(mainGPU "tests/main.cpp" ${Includes} ${Sources} ${CPUSources} ${GPUSources})
target_compile_definitions(mainGPU PRIVATE -DGPU)
target_include_directories(mainGPU PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mainGPU ${OpenCL_LIBRARY})

add_executable(0s01c-VectorAddNodeSingleCPU "tests/0s01-VectorAddNodeSingle.cpp" ${Includes} ${Sources} ${CPUSources})
target_compile_definitions(0s01c-VectorAddNodeSingleCPU PRIVATE -DCPU)
add_executable(0s02c-InputNodeSingleCPU "tests/0s02-InputNodeSingle.cpp" ${Includes} ${Sources} ${CPUSources})
target_compile_definitions(0s02c-InputNodeSingleCPU PRIVATE -DCPU)

add_executable(0s01g-VectorAddNodeSingleGPU "tests/0s01-VectorAddNodeSingle.cpp" ${Includes} ${Sources} ${GPUSources})
target_compile_definitions(0s01g-VectorAddNodeSingleGPU PRIVATE -DGPU)
target_include_directories(0s01g-VectorAddNodeSingleGPU PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(0s01g-VectorAddNodeSingleGPU ${OpenCL_LIBRARY})
add_executable(0s02g-InputNodeSingleGPU "tests/0s02-InputNodeSingle.cpp" ${Includes} ${Sources} ${GPUSources})
target_compile_definitions(0s02g-InputNodeSingleGPU PRIVATE -DGPU)
target_include_directories(0s02g-InputNodeSingleGPU PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(0s02g-InputNodeSingleGPU ${OpenCL_LIBRARY})
