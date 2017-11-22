cmake_minimum_required (VERSION 3.1)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(mpp_project)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -Wall -Werror")

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
# overall format: <milestone number><s(ingle)|g(raph)><c(orrectness)|p(erformance)><test number><c(pu)|g(pu)>
add_executable(0sc01c_VectorAddNodeSingle "tests/0sc01-VectorAddNodeSingle.cpp")
target_compile_definitions(0sc01c_VectorAddNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc01c_VectorAddNodeSingle cgLibCPU)
add_executable(0sc02c_InputNodeSingle "tests/0sc02-InputNodeSingle.cpp")
target_compile_definitions(0sc02c_InputNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc02c_InputNodeSingle cgLibCPU)
add_executable(0sc03c_MatrixMultNodeSingle "tests/0sc03-MatrixMultNodeSingle.cpp")
target_compile_definitions(0sc03c_MatrixMultNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc03c_MatrixMultNodeSingle cgLibCPU)

add_executable(0sp01c_VectorAddNodeSingle "tests/0sp01-VectorAddNodeSingle.cpp")
target_compile_definitions(0sp01c_VectorAddNodeSingle PRIVATE -DCPU)
target_include_directories(0sp01c_VectorAddNodeSingle PRIVATE ${PAPI_INCLUDE_DIRS})
target_link_libraries(0sp01c_VectorAddNodeSingle cgLibCPU ${PAPI_LIBRARIES})
add_executable(0sp03c_MatrixMultNodeSingle "tests/0sp03-MatrixMultNodeSingle.cpp")
target_compile_definitions(0sp03c_MatrixMultNodeSingle PRIVATE -DCPU)
target_include_directories(0sp03c_MatrixMultNodeSingle PRIVATE ${PAPI_INCLUDE_DIRS})
target_link_libraries(0sp03c_MatrixMultNodeSingle cgLibCPU ${PAPI_LIBRARIES})

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

add_executable(0sp01g_VectorAddNodeSingle "tests/0sp01-VectorAddNodeSingle.cpp")
target_compile_definitions(0sp01g_VectorAddNodeSingle PRIVATE -DGPU)
target_include_directories(0sp01g_VectorAddNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS} ${PAPI_INCLUDE_DIRS})
target_link_libraries(0sp01g_VectorAddNodeSingle ${OpenCL_LIBRARY} cgLibGPU ${PAPI_LIBRARIES})
add_executable(0sp03g_MatrixMultNodeSingle "tests/0sp03-MatrixMultNodeSingle.cpp")
target_compile_definitions(0sp03g_MatrixMultNodeSingle PRIVATE -DGPU)
target_include_directories(0sp03g_MatrixMultNodeSingle PRIVATE ${OpenCL_INCLUDE_DIRS} ${PAPI_INCLUDE_DIRS})
target_link_libraries(0sp03g_MatrixMultNodeSingle ${OpenCL_LIBRARY} cgLibGPU ${PAPI_LIBRARIES})
