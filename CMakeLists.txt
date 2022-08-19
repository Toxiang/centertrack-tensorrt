cmake_minimum_required(VERSION 3.5)
project(ctTrack_trt)

SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -Wno-dev")

SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall ")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE Release)

set(CUDA_ARCH64_PATH /usr/local/cuda-10.2/targets/aarch64-linux/include)
message(STATUS "cuda aarch64 path is <<<<<<<<<<", ${CUDA_ARCH64_PATH})

include_directories(${CUDA_ARCH64_PATH})

set(GPU_ARCHS 72)  ## config your GPU_ARCHS
set(TENSORRT_ROOT /usr/src/tensorrt)
set(CUDA_DIR "/usr/local/cuda-10.2")
## build
add_subdirectory(onnx-tensorrt)
add_subdirectory(track)
add_subdirectory(_main_)
