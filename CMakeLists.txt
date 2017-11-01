cmake_minimum_required (VERSION 3.1)

project(mpp_project)

set (CMAKE_CXX_STANDARD 11)

# Collect source files
file(GLOB_RECURSE Includes "includes/*hpp")
file(GLOB_RECURSE Sources "src/*.cpp")
file(GLOB_RECURSE CPUSources "src-cpu/*.cpp")
file(GLOB_RECURSE GPUSources "src-gpu/*.cpp")
include_directories("includes")

option(GPUImplementation "Compile with GPU accelerated OpenCL implementation of compute graph evaluation. Turn off for sequential CPU evaluation." OFF)

if(GPUImplementation)
    add_definitions(-DGPU)
    # Handle OpenCL
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS})
    link_directories(${OpenCL_LIBRARY})
else(GPUImplementation)
    add_definitions(-DCPU)
endif(GPUImplementation)

add_executable(main ${Sources} ${CPUSources} ${GPUSources})
if(GPUImplementation)
    target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(main ${OpenCL_LIBRARY})
endif(GPUImplementation)
