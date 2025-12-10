cmake_minimum_required(VERSION 3.26)

project(backend)

# Project-wide setup.
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

enable_language("CUDA")
set(CMAKE_CUDA_ARCHITECTURES "native")
# Warnings suppressed in the lastest Eigen. When using an old version we manually suppress them here.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=815")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20011")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20012")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20013")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20014")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20015")

# Configure include and lib paths.
include_directories("${ROOT_DIR}/")
include_directories("${ROOT_DIR}/cpp/")
include_directories("${ROOT_DIR}/external/eigen/")

# Compile the python binding library.
include_directories("${ROOT_DIR}/external/pybind11/include/")
add_subdirectory("${ROOT_DIR}/external/pybind11/" "${ROOT_DIR}/build/pybind11")

file(GLOB_RECURSE binding_cpp_files "./*.cpp" "./*.cu")
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
pybind11_add_module(backend ${binding_cpp_files})