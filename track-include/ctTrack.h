// 
// write by Toxic in 2021.5.15
// 晚上21.43
// 

#ifndef CTTRACK_TRT_CTTRACK_H
#define CTTRACK_TRT_CTTRACK_H

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "ctTrackConfig.h"
#include "utils.h"
//#include "NvOnnxParserRuntime.h"

namespace Track
{
    //float32
    enum class MODE
    {
        FLOAT32 = 0,
        FLOAT16 = 1,
        INT8 = 2
    };
    class ctTrack
    {
    public:
        // 构造函数
        ctTrack(const std::string& onnxFile,const std::string& calibFile,MODE mode = MODE::FLOAT32);
        ctTrack(const std::string& engineFile);
        ~ctTrack(){
            cudaStreamSynchronize(mCudaStream);
            cudaStreamDestroy(mCudaStream);
            for(auto& item : mCudaBuffers)
                cudaFree(item);
            cudaFree(cudaOutputBuffer);
            if(!mRunTime)
                mRunTime->destroy();
            if(!mContext)
                mContext->destroy();
            if(!mEngine)
                mEngine->destroy();
            if(!mPlugins)
                mPlugins->destroy();
        }
        
        void saveEngine(const std::string& fileName);
        std::vector<Detection> processDet ();
        std::vector<Detection> postProcessDet();
        void doInference2(const void *cur_img,const void *pre_img,const void *pre_hm,
							  void* res_img);
        void doInference(const void *cur_img,const void *pre_img,const void *pre_hm,std::vector<Detection>& res_img,cv::Mat& img);
        std::vector<int> sort_index(const std::vector<float> &v);
        void printTime(){
            mProfiler.printTime(runItems);
        }

        inline size_t getInputSize(){
            return mBindBufferSizes[0];
        };

        int64_t outputBufferSize;
        bool flag;
    private:
        void initEngine();
        nvinfer1::IExecutionContext* mContext;
        nvinfer1::ICudaEngine* mEngine;
        nvinfer1::IRuntime* mRunTime;

        MODE Mode;

        nvonnxparser::IPluginFactory* mPlugins;
        void *mCudaBuffers[7];
        std::vector<int64_t>mBindBufferSizes;
        void * cudaOutputBuffer;
        int outSize1,outSize2,outSize3,outSize4;
        float *out1 ;
        float *out2 ;
        float *out3 ;
        float *out4;
        // std::vector<int64_t> mBindBufferSizes;
        cudaStream_t mCudaStream;

        int runItems;
        Profiler mProfiler;
    };
    
}

#endif