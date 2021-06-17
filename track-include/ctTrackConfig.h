// 
// First create by Toxic 2021.5.15
//

// 网络模型的参数配置 

#ifndef CTTRACK_TRT_CTTRACKCONFIG_H
#define CTTRACK_TRT_CTTRACKCONFIG_H

namespace Track{
    //阈值
    constexpr static float visThresh = xxx;
    //maxpool size
    constexpr static int kernelSize = xxx;

    //网络参数
    constexpr static int input_h = xxx;
    constexpr static int input_w = xxx;
    constexpr static int channel = xxx;
    constexpr static int classNum = xxx;
    constexpr static float mean[] = {xxx, xxx, xxx};
    constexpr static float std[] = {xxx, xxx, xxx};
    constexpr static char *className[]= {(char*)"xxx",(char*)"xxx",(char*)"xxx"};


}

#endif 
