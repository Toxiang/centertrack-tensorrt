// 
// First create by Toxic 2021.5.15
//

// 网络模型的参数配置 

#ifndef CTTRACK_TRT_CTTRACKCONFIG_H
#define CTTRACK_TRT_CTTRACKCONFIG_H

namespace Track{

    constexpr static float visThresh = 0.3; // threshold for visualization
    constexpr static int kernelSize = 3 ;  /// nms maxpool size
    //网络参数, FOR 3D TRACKING
   constexpr static int input_w = 800 ;
    constexpr static int input_h = 448 ;
    constexpr static int channel = 3 ;
    constexpr static int classNum = 3 ;
    constexpr static float mean[]= {0.485, 0.456, 0.406};
    constexpr static float std[] = {0.229, 0.224, 0.225};
    constexpr static char *className[]= {(char*)"pedestrian", (char*)"car", (char*)"cyclist"};


}

#endif 
