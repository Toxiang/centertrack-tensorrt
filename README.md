# centertrack-tensorrt
将centertrack部署到tensorrt上


从https://github.com/CaoWGG/TensorRT-CenterNet 改写过来
对tensorrt和centertrack了解一般般 
只是能跑通 效果也一般般

onnx-tensorrt文件夹从https://github.com/CaoWGG/TensorRT-CenterNet 下载

environment:
  ubuntu 16.0.4
  pytorch 1.2
  tensorrt 5
  protobuf 3.15
  (反正尽量跟着centernet-tensorrt走就是了）
  
ex.:
注意：
  其中在track/tracking.cpp 下需要修改参数 适应自己的
        track-include/ctTrackConfig 下修改模型参数
        还有些参数 适应自己的就好
  mkdir build
  cd build && cmake .. && make -j3
  ./buildEngine -i your_onnx_model_path -o output_engine_file
  ./tracking -e engine_file_path -i xx.jpg
  

