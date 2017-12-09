cmake_minimum_required (VERSION 3.1)

project(cglib-tests LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CMAKE_CURRENT_SOURCE_DIR}/mnist")

find_package(PAPI)
if (NOT ${PAPI_FOUND})
    message("NOTICE: PAPI couldn't be found. Performance test executables will not be built. Try setting PAPI_PREFIX to the install directory of PAPI.")
endif (NOT ${PAPI_FOUND})

find_package(MNIST)
if (NOT ${MNIST_FOUND})
    message("NOTICE: MNIST loader library couldn't be found. MNIST classifier executable will not be built.")
endif (NOT ${MNIST_FOUND})

##########################################################################################################################
############ PLAYGROUND TESTBED: tests/main.cpp is a general playground for in-development testing #######################
##########################################################################################################################
add_executable(mainCPU "main.cpp")
target_compile_definitions(mainCPU PRIVATE -DCPU)
target_link_libraries(mainCPU cgLibCPU)

add_executable(mainGPU "main.cpp")
target_compile_definitions(mainGPU PRIVATE -DGPU)
target_link_libraries(mainGPU cgLibGPU)


##########################################################################################################################
################# SINGLE NODE TESTS: milestone 1, testing the implementation of single nodes #############################
##########################################################################################################################

# naming convention: 0sc01c is read as : 0s -> stage 0, single node test, c -> correctness test, 01 -> counter, c -> cpu implementation test
#                    0sc01g is read as : 0s -> stage 0, single node test, c -> correctness test, 01 -> counter, g -> gpu implementation test
#                    0sp01c is read as : 0s -> stage 0, single node test, p -> performance test, 01 -> counter, c -> cpu implementation test
# overall format: <milestone number><s(ingle)|g(raph)><c(orrectness)|p(erformance)><test number><c(pu)|g(pu)>

###### Correctness tests on CPU
add_executable(0sc01c_VectorAddNodeSingle "0sc01-VectorAddNodeSingle.cpp")
target_compile_definitions(0sc01c_VectorAddNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc01c_VectorAddNodeSingle cgLibCPU)
add_executable(0sc02c_InputNodeSingle "0sc02-InputNodeSingle.cpp")
target_compile_definitions(0sc02c_InputNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc02c_InputNodeSingle cgLibCPU)
add_executable(0sc03c_MatrixMultNodeSingle "0sc03-MatrixMultNodeSingle.cpp")
target_compile_definitions(0sc03c_MatrixMultNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc03c_MatrixMultNodeSingle cgLibCPU)
add_executable(0sc04c_VectorDivNodeSingle "0sc04-VectorDivNodeSingle.cpp")
target_compile_definitions(0sc04c_VectorDivNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc04c_VectorDivNodeSingle cgLibCPU)
add_executable(0sc05c_VectorMultNodeSingle "0sc05-VectorMultNodeSingle.cpp")
target_compile_definitions(0sc05c_VectorMultNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc05c_VectorMultNodeSingle cgLibCPU)
add_executable(0sc06c_ReduceMeanNodeSingle "0sc06-ReduceMeanNodeSingle.cpp")
target_compile_definitions(0sc06c_ReduceMeanNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc06c_ReduceMeanNodeSingle cgLibCPU)
add_executable(0sc07c_SliceNodeSingle "0sc07-SliceNodeSingle.cpp")
target_compile_definitions(0sc07c_SliceNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc07c_SliceNodeSingle cgLibCPU)
add_executable(0sc08c_StackNodeSingle "0sc08-StackNodeSingle.cpp")
target_compile_definitions(0sc08c_StackNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc08c_StackNodeSingle cgLibCPU)
add_executable(0sc09c_TranposeNodeSingle "0sc09-TransposeNodeSingle.cpp")
target_compile_definitions(0sc09c_TranposeNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc09c_TranposeNodeSingle cgLibCPU)
add_executable(0sc10c_ExpFuncNodeSingle "0sc10-ExpFuncNodeSingle.cpp")
target_compile_definitions(0sc10c_ExpFuncNodeSingle PRIVATE -DCPU)
target_link_libraries(0sc10c_ExpFuncNodeSingle cgLibCPU)

###### Correctness tests on GPU
add_executable(0sc01g_VectorAddNodeSingle "0sc01-VectorAddNodeSingle.cpp")
target_compile_definitions(0sc01g_VectorAddNodeSingle PRIVATE -DGPU)
target_link_libraries(0sc01g_VectorAddNodeSingle cgLibGPU)
add_executable(0sc02g_InputNodeSingle "0sc02-InputNodeSingle.cpp")
target_compile_definitions(0sc02g_InputNodeSingle PRIVATE -DGPU)
target_link_libraries(0sc02g_InputNodeSingle cgLibGPU)
add_executable(0sc03g_MatrixMultNodeSingle "0sc03-MatrixMultNodeSingle.cpp")
target_compile_definitions(0sc03g_MatrixMultNodeSingle PRIVATE -DGPU)
target_link_libraries(0sc03g_MatrixMultNodeSingle cgLibGPU)
add_executable(0sc04g_VectorDivNodeSingle "0sc04-VectorDivNodeSingle.cpp")
target_compile_definitions(0sc04g_VectorDivNodeSingle PRIVATE -DGPU)
target_link_libraries(0sc04g_VectorDivNodeSingle cgLibGPU)
add_executable(0sc05g_VectorMultNodeSingle "0sc05-VectorMultNodeSingle.cpp")
target_compile_definitions(0sc05g_VectorMultNodeSingle PRIVATE -DGPU)
target_link_libraries(0sc05g_VectorMultNodeSingle cgLibGPU)
add_executable(0sc06g_ReduceMeanNodeSingle "0sc06-ReduceMeanNodeSingle.cpp")
target_compile_definitions(0sc06g_ReduceMeanNodeSingle PRIVATE -DGPU)
target_link_libraries(0sc06g_ReduceMeanNodeSingle cgLibGPU)
add_executable(0sc07g_SliceNodeSingle "0sc07-SliceNodeSingle.cpp")
target_compile_definitions(0sc07g_SliceNodeSingle PRIVATE -DGPU)
target_link_libraries(0sc07g_SliceNodeSingle cgLibGPU)
add_executable(0sc08g_StackNodeSingle "0sc08-StackNodeSingle.cpp")
target_compile_definitions(0sc08g_StackNodeSingle PRIVATE -DGPU)
target_link_libraries(0sc08g_StackNodeSingle cgLibGPU)
add_executable(0sc09g_TranposeNodeSingle "0sc09-TransposeNodeSingle.cpp")
target_compile_definitions(0sc09g_TranposeNodeSingle PRIVATE -DGPU)
target_link_libraries(0sc09g_TranposeNodeSingle cgLibGPU)
add_executable(0sc10g_ExpFuncNodeSingle "0sc10-ExpFuncNodeSingle.cpp")
target_compile_definitions(0sc10g_ExpFuncNodeSingle PRIVATE -DGPU)
target_link_libraries(0sc10g_ExpFuncNodeSingle cgLibGPU)

if (${PAPI_FOUND})
    ###### Performance tests on CPU
    add_executable(0sp01c_VectorAddNodeSingle "0sp01-VectorAddNodeSingle.cpp")
    target_compile_definitions(0sp01c_VectorAddNodeSingle PRIVATE -DCPU)
    target_include_directories(0sp01c_VectorAddNodeSingle PRIVATE ${PAPI_INCLUDE_DIRS})
    target_link_libraries(0sp01c_VectorAddNodeSingle cgLibCPU ${PAPI_LIBRARIES})
    add_executable(0sp03c_MatrixMultNodeSingle "0sp03-MatrixMultNodeSingle.cpp")
    target_compile_definitions(0sp03c_MatrixMultNodeSingle PRIVATE -DCPU)
    target_include_directories(0sp03c_MatrixMultNodeSingle PRIVATE ${PAPI_INCLUDE_DIRS})
    target_link_libraries(0sp03c_MatrixMultNodeSingle cgLibCPU ${PAPI_LIBRARIES})

    ###### Performance tests on GPU
    add_executable(0sp01g_VectorAddNodeSingle "0sp01-VectorAddNodeSingle.cpp")
    target_compile_definitions(0sp01g_VectorAddNodeSingle PRIVATE -DGPU)
    target_include_directories(0sp01g_VectorAddNodeSingle PRIVATE ${PAPI_INCLUDE_DIRS})
    target_link_libraries(0sp01g_VectorAddNodeSingle cgLibGPU ${PAPI_LIBRARIES})
    add_executable(0sp03g_MatrixMultNodeSingle "0sp03-MatrixMultNodeSingle.cpp")
    target_compile_definitions(0sp03g_MatrixMultNodeSingle PRIVATE -DGPU)
    target_include_directories(0sp03g_MatrixMultNodeSingle PRIVATE ${PAPI_INCLUDE_DIRS})
    target_link_libraries(0sp03g_MatrixMultNodeSingle cgLibGPU ${PAPI_LIBRARIES})
endif (${PAPI_FOUND})



##########################################################################################################################
######################## GRAPH TESTS: milestone 2, testing the implementation of a full graph ############################
##########################################################################################################################
###### Correctness tests on CPU
add_executable(1gc01c_Graph "1gc01-Graph.cpp")
target_compile_definitions(1gc01c_Graph PRIVATE -DCPU)
target_include_directories(1gc01c_Graph PRIVATE ${MNIST_INCLUDE_DIR})
target_link_libraries(1gc01c_Graph cgLibCPU)

###### Correctness tests on GPU
add_executable(1gc01g_Graph "1gc01-Graph.cpp")
target_compile_definitions(1gc01g_Graph PRIVATE -DGPU)
target_include_directories(1gc01g_Graph PRIVATE ${MNIST_INCLUDE_DIR})
target_link_libraries(1gc01g_Graph cgLibGPU)

if (${MNIST_FOUND} AND ${PAPI_FOUND})
    ###### Performance tests on CPU
    add_executable(1gp01c_Graph "1gp01-Graph.cpp")
    target_compile_definitions(1gp01c_Graph PRIVATE -DCPU)
    target_include_directories(1gp01c_Graph PRIVATE ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
    target_link_libraries(1gp01c_Graph  cgLibCPU ${PAPI_LIBRARIES})
    add_executable(1gp02c_MNISTGraphForwardOnly "1gp02-MNISTGraphForwardOnly.cpp")
    target_compile_definitions(1gp02c_MNISTGraphForwardOnly PRIVATE -DCPU)
    target_include_directories(1gp02c_MNISTGraphForwardOnly PRIVATE ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
    target_link_libraries(1gp02c_MNISTGraphForwardOnly cgLibCPU ${PAPI_LIBRARIES})
    add_executable(1gp03c_MNISTGraph "1gp03-MNISTGraph.cpp")
    target_compile_definitions(1gp03c_MNISTGraph PRIVATE -DCPU)
    target_include_directories(1gp03c_MNISTGraph PRIVATE ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
    target_link_libraries(1gp03c_MNISTGraph cgLibGPU ${PAPI_LIBRARIES})

    ###### Performance tests on GPU
    add_executable(1gp01g_Graph "1gp01-Graph.cpp")
    target_compile_definitions(1gp01g_Graph PRIVATE -DGPU)
    target_include_directories(1gp01g_Graph PRIVATE ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
    target_link_libraries(1gp01g_Graph cgLibGPU ${PAPI_LIBRARIES})
    add_executable(1gp02g_MNISTGraphForwardOnly "1gp02-MNISTGraphForwardOnly.cpp")
    target_compile_definitions(1gp02g_MNISTGraphForwardOnly PRIVATE -DGPU)
    target_include_directories(1gp02g_MNISTGraphForwardOnly PRIVATE ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
    target_link_libraries(1gp02g_MNISTGraphForwardOnly cgLibGPU ${PAPI_LIBRARIES})
    add_executable(1gp03g_MNISTGraph "1gp03-MNISTGraph.cpp")
    target_compile_definitions(1gp03g_MNISTGraph PRIVATE -DGPU)
    target_include_directories(1gp03g_MNISTGraph PRIVATE ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
    target_link_libraries(1gp03g_MNISTGraph cgLibGPU ${PAPI_LIBRARIES})

    ###### MNIST tasks baseline/benchmark non-abstracted CPU implementation
    add_executable(1gp02b_MNISTGraphForwardOnlyBaseline "1gp02b-MNISTGraphForwardOnlyBaseline.cpp")
    target_compile_definitions(1gp02b_MNISTGraphForwardOnlyBaseline PRIVATE)
    target_include_directories(1gp02b_MNISTGraphForwardOnlyBaseline PRIVATE ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
    target_link_libraries(1gp02b_MNISTGraphForwardOnlyBaseline ${PAPI_LIBRARIES})
    add_executable(1gp03b_MNISTGraphBaseline "1gp03b-MNISTGraphBaseline.cpp")
    target_compile_definitions(1gp03b_MNISTGraphBaseline PRIVATE)
    target_include_directories(1gp03b_MNISTGraphBaseline PRIVATE ${PAPI_INCLUDE_DIRS} ${MNIST_INCLUDE_DIR})
    target_link_libraries(1gp03b_MNISTGraphBaseline ${PAPI_LIBRARIES})
endif (${MNIST_FOUND} AND ${PAPI_FOUND})
