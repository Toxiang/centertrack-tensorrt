// 
// First create by Toxic 2021.5.15
//

// 网络模型的参数配置 

#ifndef CTTRACK_TRT_CTTRACKCONFIG_H
#define CTTRACK_TRT_CTTRACKCONFIG_H

namespace Track{
    //阈值
    constexpr static float visThresh = 0.3;
    //maxpool size
    constexpr static int kernelSize = 3;

    //网络参数
    constexpr static int input_h = 384;
    constexpr static int input_w = 1280;
    constexpr static int channel = 3;
    constexpr static int classNum = 3;
    constexpr static float mean[] = {0.408, 0.447, 0.470};
    constexpr static float std[] = {0.289, 0.274, 0.278};
    constexpr static char *className[]= {(char*)"ped",(char*)"car",(char*)"cyc"};


}

#endif 