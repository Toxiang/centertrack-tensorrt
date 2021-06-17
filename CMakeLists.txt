cmake_minimum_required(VERSION 3.5)
project(ctTrack_trt)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE Release)
LINK_DIRECTORIES(/usr/local/cuda-9.0/lib64)
set(GPU_ARCHS 61)  ## config your GPU_ARCHS,See [here](https://developer.nvidia.com/cuda-gpus) for finding what maximum compute capability your specific GPU supports.
set(TENSORRT_ROOT /home/wangzx/下载/TensorRT-5.0.2.6)

## build
add_subdirectory(onnx-tensorrt)
add_subdirectory(track)
add_subdirectory(_main_)
