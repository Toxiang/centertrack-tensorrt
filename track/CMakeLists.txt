cmake_minimum_required(VERSION 3.5)
project(ctTrack_trt)
find_package(CUDA REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
message(STATUS "Found TensorRT libs ${TENSORRT_LIBRARY_INFER}")

find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARIES_DIRS})

file(GLOB CPP_SRC *.cpp)
file(GLOB CU_SRC *.cu)

file(GLOB log_src ../track-include/common/logger.cpp)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast")
list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")

include_directories(${CUDA_INCLUDE_DIRS}
                    ${TENSORRT_INCLUDE_DIR}
                    ${OpenCV_INCLUDE_DIRS}
                    ../track-include
                    ../onnx-tensorrt
                    ../plugins/include
                    ../plugins/parsers/common
                    ../plugins/onnx
                    ../plugins/plugin
                    ../plugins/plugin/common)

  set(PLUGIN_LISTS
                    ../plugins/plugin/DCNv2
                    #../plugins/plugin/instanceNormalizationPlugin
                    )
                    
include_directories(../plugins/plugin/common 
                    ../plugins/plugin/common/kernels 
                    ../plugins/samples/common)


                foreach(PLUGIN_ITER ${PLUGIN_LISTS})
                    #message(STATUS "PLUGIN_ITER:${PLUGIN_ITER} \n")
                    include_directories(${PLUGIN_ITER})
                    #file(GLOB SRC_CU_PLUGINS ${PLUGIN_ITER}/*cu)
                    #file(GLOB SRC_CPP_PLUGINS ${PLUGIN_ITER}/*cpp)
                    #list(APPEND ALL_SRCS "${PLUGIN_ITER}/*.cu")     
                endforeach(PLUGIN_ITER)
                
                file(GLOB ALL_SRCS 
                    ../plugins/plugin/DCNv2/*.cpp
                    ../plugins/plugin/InferPlugin.cpp    
                    #../plugins/plugin/instanceNormalizationPlugin/*.cpp
                
                
                    ../plugins/plugin/DCNv2/*.cu
        
                    )                   

cuda_add_library(Track SHARED ${ALL_SRCS} ${CPP_SRC} ${CU_SRC} ${log_src})
target_link_libraries(Track
        ${TENSORRT_LIBRARY_INFER}
        ${OpenCV_LIBS}
        nvonnxparser
        )
