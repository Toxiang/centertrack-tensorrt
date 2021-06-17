#include"trackForwardGPU.h"
#include"utils.h"

dim3 cudaGridSize(uint n)
{
    uint k = (n - 1) /BLOCK + 1;
    uint x = k ;
    uint y = 1 ;
    if (x > 65535 )
    {
        x = ceil(sqrt(x));
        y = (n - 1 )/(x*BLOCK) + 1;
    }
    dim3 d = {x,y,1} ;
    return d;
}

__device__ float sigmoid(float data){ return 1./(1. + exp(-data)); }

__global__ void ctTrack_Forward_GPU_kernel(const float *hm,const float *reg,const float* wh,float *output,
                         const int w,const int h,const int classes,const int kernel_size,
                         const float visthresh)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= w * h * classes) return;
    int padding = 1;
    int offset = -padding;
    int stride = w * h;
    int x = index % w;
    int y = (index / w) % h;
    int classN = index / w / h ;
    int regIndex = index - classN*stride;
    float c_x, c_y;
    float objProb = sigmoid(hm[index]);
    if (objProb > visthresh) {
        float max = -1;
        int maxIndex = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; ++j) {
                int xx = offset+i+x;
                int curX = (xx>0)?xx:0;
                curX = (curX<w)?curX:w-1;
                int yy = offset+j+y;
                int curY = (yy>0)?yy:0;
                curY = (curY<h)?curY:h-1;
                int curIndex = curY*w+curX+stride*classN;
                float val = sigmoid(hm[curIndex]);
                maxIndex = (val>max)?curIndex:maxIndex;
                max = (val>max)?val:max;
            }

        if(index == maxIndex){
            int resCount = (int) atomicAdd(output, 1);
            //printf("%d",resCount);
            char *data = (char *) output + sizeof(float) + resCount * sizeof(Detection);
            Detection *det = (Detection *) (data);
            c_x = x + reg[regIndex];
            c_y = y + reg[regIndex + stride];
            det->bbox.x1 = (c_x - wh[regIndex] / 2) * 4;
            det->bbox.y1 = (c_y - wh[regIndex + stride] / 2) * 4;
            det->bbox.x2 = (c_x + wh[regIndex] / 2) * 4;
            det->bbox.y2 = (c_y + wh[regIndex + stride] / 2) * 4;
            det->classId = classN;
            det->prob = objProb;
        }
    }

}

void ctTrack_Forward_GPU(const float *hm, const float *reg,const float *wh ,float *output,
                      const int w,const int h,const int classes,const int kernerl_size, const float visthresh ){
    uint num = w * h * classes;
    ctTrack_Forward_GPU_kernel<<<cudaGridSize(num),BLOCK>>>(hm,reg,wh,output,w,h,classes,kernerl_size,visthresh);
}