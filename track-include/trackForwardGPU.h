//
// First create by Toxic 2021.5.15
// 

#ifndef CTTRACK_TRT_TRACKFORWARDGPU_H
#define CTTRACK_TRT_TRACKFORWARDGPU_H
void ctTrack_Forward_GPU(const float *hm,const float *reg,const float* wh,float *output,
                         const int w,const int h,const int classes,const int kernel_size,
                         const float visthresh);


#endif