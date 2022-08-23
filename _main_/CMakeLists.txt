cmake_minimum_required(VERSION 3.5)
project(_main_)

SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast")
include_directories(../track-include
                    ../onnx-tensorrt)

add_executable(buildEngine buildEngine.cpp)
target_link_libraries(buildEngine Track)

add_executable(tracking tracking.cpp)
target_link_libraries(tracking Track)
