# TensorRT8-DCNv2-Plugin
基于TensorRT8实现DCNv2插件

## prerequirements
1. TensorRT-8.2GA

## dependencies
1. TensorRT 8.2GA
2. onnx-tensorrt corresponding to RT8.2

1. clone TensorRT release/8.2版本
2. 将DCNv2文件夹、InferPlugin.cpp以及CMakeLists拷贝到TensorRT/plugin中
3. 在TensorRT目录创建build目录，进入后运行
```
mkdir build && cd build
cmake .. -DBUILD_PARSERS=OFF -DBUILD_SAMPLES=OFF -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_BIN_DIR=`pwd`/out -DBUILD_PLUGINS=ON -DCUB_ROOT_DIR=$CUB_PATH
make -j4
```
4. 编译完成后会生成libnvinfer_plugin库
5. 将builtin_op_importer.cpp拷贝到onnx-tensorrt中编译libnvonnxparser库

DCNv2的实现代码摘自CaoWGG/TensorRT-CenterNet
