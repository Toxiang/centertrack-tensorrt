#ifndef CTTRACK_TRT_UTILS_H
#define CTTRACK_TRT_UTILS_H

#include<map>
#include<iostream>
#include<iomanip>
#include<iostream>
#include<assert.h>
#include<numeric>
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include "cuda.h"
#include "cuda_runtime.h"

#ifndef BLOCK
#define BLOCK 512
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }
#endif

class Profiler:public nvinfer1::IProfiler
{
public:
    struct Record
    {
        float time{0};
        int count{0};
    };

    void printTime(const int& runTimes)
    {
        float totalTime = 0;
        std::string layerNameStr = "TRT Layer Name:";
        int maxLayerNameLength = std::max(static_cast<int>(layerNameStr.size()),70);
        for(const auto& elem:mProfile)
        {
            totalTime += elem.second.time;
            maxLayerNameLength = std::max(maxLayerNameLength,static_cast<int>(elem.first.size()));
        }
        std::cout<<"total time = "<<totalTime/runTimes<<"ms"<<std::endl;
    }

    virtual void reportLayerTime(const char* layerName,float ms) noexcept
    {
        mProfile[layerName].count++;
        mProfile[layerName].time += ms;
    }
private:
    std::map<std::string,Record> mProfile;
};

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
            : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) noexcept
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    };
    
    Severity reportableSeverity;
};

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CUDA_CHECK(cudaMalloc(&deviceMem,memSize));
    if(deviceMem==nullptr)
    {
        std::cerr<<"Out of memory"<<std::endl;
        exit(1);
    }
    return deviceMem;
}

struct Box{
    float x1;
    float x2;
    float y1;
    float y2;
};

struct Detection{
    Box bbox;
    int classId;
    float prob;
    float tracking_x;
    float tracking_y;
};

extern std::vector<float>prepareImage(cv::Mat& img);
extern void postProcess(std::vector<Detection>&result,const cv::Mat& img);
extern void drawImg(const std::vector<Detection>& result,cv::Mat& img,const std::vector<cv::Scalar>& color);
extern cv::Scalar randomColor(cv::RNG& rng);

#endif