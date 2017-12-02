cmake_minimum_required (VERSION 3.1)

project(cglib LANGUAGES CXX)

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Werror")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -Wall -Werror")

# Collect source files
file(GLOB_RECURSE Includes "includes/*hpp")
file(GLOB_RECURSE Sources "src/*.cpp" "src/*.hpp")
file(GLOB_RECURSE CPUSources "src-cpu/*.cpp" "src-cpu/*.hpp")
file(GLOB_RECURSE GPUSources "src-gpu/*.cpp" "src-gpu/*.hpp")

##########################################################################################################################
############################################### GRAPH COMPUTATION LIBRARY ################################################
##########################################################################################################################

add_library(cgLibCPU SHARED ${Includes} ${Sources} ${CPUSources})
target_compile_definitions(cgLibCPU PRIVATE -DCPU)
target_include_directories(cgLibCPU PUBLIC "${PROJECT_SOURCE_DIR}/includes")

add_library(cgLibGPU SHARED ${Includes} ${Sources} ${GPUSources} ${OCLKernelSources})
target_compile_definitions(cgLibGPU PRIVATE -DGPU)
target_include_directories(cgLibGPU PUBLIC "${PROJECT_SOURCE_DIR}/includes" ${OpenCL_INCLUDE_DIRS})
target_link_libraries(cgLibGPU PUBLIC ${OpenCL_LIBRARY})
