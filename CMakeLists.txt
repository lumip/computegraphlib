#Minimal OpenCL CMakeLists.txt by StreamHPC

cmake_minimum_required (VERSION 3.1)

project(mpp_project)

set (CMAKE_CXX_STANDARD 11)

# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

# Collect source files
file(GLOB_RECURSE Sources "src/*.cpp")
include_directories("src")

add_executable(main ${Sources})
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(main ${OpenCL_LIBRARY})