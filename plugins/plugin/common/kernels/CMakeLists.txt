#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
file(GLOB SRCS *.cpp)
set(PLUGIN_SOURCES ${PLUGIN_SOURCES} ${SRCS})
set(PLUGIN_SOURCES ${PLUGIN_SOURCES} PARENT_SCOPE)

if(NOT CUB_ROOT_DIR)
    set(CUB_ROOT_DIR /home/nvidia/xuewei/3D/plugins/plugin/cub/)
    message(STATUS "SET CUB_PATH" ${CUB_ROOT_DIR})
    #set(CUB_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cub CACHE STRING "directory of CUB installation")
endif()

message(STATUS " CUB_PATH IS:<<<<<<<<<<<<wilson:" ${CUB_ROOT_DIR})
#INCLUDE_DIRECTORIES(${CUB_ROOT_DIR})
#target_link_directories(${CUB_ROOT_DIR})
include_directories(${CUB_ROOT_DIR})
file(GLOB CU_SRCS *.cu)
set(PLUGIN_CU_SOURCES ${PLUGIN_CU_SOURCES} ${CU_SRCS})
set(PLUGIN_CU_SOURCES ${PLUGIN_CU_SOURCES} PARENT_SCOPE)
