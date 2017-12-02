cmake_minimum_required (VERSION 3.1)

project(cglib LANGUAGES CXX)

# Handle OpenCL
find_package(OpenCL REQUIRED)

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

add_library(cgLibGPU SHARED ${Includes} ${Sources} ${GPUSources})
target_compile_definitions(cgLibGPU PRIVATE -DGPU)
target_include_directories(cgLibGPU PUBLIC "${PROJECT_SOURCE_DIR}/includes" ${OpenCL_INCLUDE_DIRS})
target_link_libraries(cgLibGPU PUBLIC ${OpenCL_LIBRARY})
