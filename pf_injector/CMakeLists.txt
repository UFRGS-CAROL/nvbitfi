cmake_minimum_required(VERSION 3.16)
project(pf_injector LANGUAGES CXX CUDA)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
enable_language(CUDA)
set(CUDA_HOST_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(
        ../common
        ../../../core
        ../../NVBit/nvbit_release/core

        /usr/local/cuda/include
)
add_library(pf_injector inject_funcs.cu pf_injector.cu)

set_property(TARGET pf_injector PROPERTY CUDA_ARCHITECTURES 35 60 86)

set_target_properties(
        pf_injector
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(pf_injector LogHelper)