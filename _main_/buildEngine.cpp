// 
// Toxic 2021 5.17 15:33
// 

#include <argparse.h>
#include <string>
#include <iostream>
#include "ctTrack.h"
#include "utils.h"

int main(int argc,const char** argv)
{
    optparse::OptionParser parser;
    parser.add_option("-i","--input-onnx-file").dest("onnxFile").help("onnx文件位置");
    parser.add_option("-o","--output-engine-file").dest("outputFile").help("engine文件位置");
    parser.add_option("-m","--mode").dest("mode").set_default<int>(0).help("MODE 默认时float32");
    parser.add_option("-c","--calib").dest("calibFile").help("calib文件");
    optparse::Values options = parser.parse_args(argc,argv);
    if(options["onnxFile"].size()==0){
        std::cout<<"------------未输入onnx文件-----------------"<<std::endl;
        exit(-1);
    }
    Track::MODE mode = Track::MODE::FLOAT32;
    if(options["mode"]=="0") mode = Track::MODE::FLOAT32;

    Track::ctTrack track(options["onnxFile"],options["calibFile"],mode);
    track.saveEngine(options["outputFile"]);

    std::cout<<"----------引擎保存成功！------------"<<std::endl;
}
