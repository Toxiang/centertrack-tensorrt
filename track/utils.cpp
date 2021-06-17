// 
// Toxic code in 2021.5.17 
// 

#include"utils.h"
#include<sstream>
#include"ctTrackConfig.h"

std::vector<float> prepareImage(cv::Mat& img){
    int c = Track::channel;
    int w = Track::input_w;
    int h = Track::input_h;
    float scale = cv::min(float(w)/img.cols,float(h)/img.rows);
    auto scaleSize = cv::Size(img.cols*scale,img.rows*scale);

    cv::Mat resized;
    cv::resize(img,resized,scaleSize,0,0);

    cv::Mat cropped = cv::Mat::zeros(h,w,CV_8UC3);
    cv::Rect rect((w-scaleSize.width)/2,(h-scaleSize.height)/2,scaleSize.width,scaleSize.height);

    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    cropped.convertTo(img_float,CV_32FC3,1./255.);

    // HWC to CHW
    std::vector<cv::Mat>input_c(c);
    cv::split(img_float,input_c);


    //归一化
    std::vector<float>res_img(h*w*c);
    auto data = res_img.data();
    int CL = h*w;
    for(int i=0;i<c;i++){
        cv::Mat normed_c = (input_c[i]-Track::mean[i])/Track::std[i];
        memcpy(data,normed_c.data,CL*sizeof(float));
        data+=CL;
    }
    return res_img;
}

cv::Scalar randomColor(cv::RNG& rng) {
    int icolor = (unsigned) rng;
    return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

void postProcess(std::vector<Detection>&result,const cv::Mat& img){
    using namespace cv;
    int mark;
    int input_w = Track::input_w;
    int input_h = Track::input_h;
    float scale = min(float(input_w)/img.cols,float(input_h)/img.rows);
    float dx = (input_w - scale * img.cols) / 2;
    float dy = (input_h - scale * img.rows) / 2;
    for(auto&item:result)
    {
        float x1 = (item.bbox.x1 - dx) / scale ;
        float y1 = (item.bbox.y1 - dy) / scale ;
        float x2 = (item.bbox.x2 - dx) / scale ;
        float y2 = (item.bbox.y2 - dy) / scale ;
        x1 = (x1 > 0 ) ? x1 : 0 ;
        y1 = (y1 > 0 ) ? y1 : 0 ;
        x2 = (x2 < img.cols  ) ? x2 : img.cols - 1 ;
        y2 = (y2 < img.rows ) ? y2  : img.rows - 1 ;
        item.bbox.x1  = x1 ;
        item.bbox.y1  = y1 ;
        item.bbox.x2  = x2 ;
        item.bbox.y2  = y2 ;
        item.tracking_x /= scale;
        item.tracking_y /= scale;
    }
}

void drawImg(const std::vector<Detection> & result,cv::Mat& img,const std::vector<cv::Scalar>& color)
{
    int mark;

    int box_think = (img.rows+img.cols) * .001 ;
    float label_scale = img.rows * 0.0009;
    int base_line ;
    std::cout << "label_scale:" << label_scale << " box_think:" << box_think << std::endl;
    for (const auto &item : result) {
        std::string label;
        std::stringstream stream;
        stream << Track::className[item.classId] << " " << item.prob << std::endl;
        std::getline(stream,label);

        auto size = cv::getTextSize(label,cv::FONT_HERSHEY_COMPLEX,label_scale,1,&base_line);

        cv::rectangle(img, cv::Point(item.bbox.x1,item.bbox.y1),
                      cv::Point(item.bbox.x2 ,item.bbox.y2),
                      color[item.classId], box_think*2, 1, 0);
        
        cv::putText(img,label,
                cv::Point(item.bbox.x2,item.bbox.y2 - size.height),
                cv::FONT_HERSHEY_COMPLEX, label_scale , color[item.classId], box_think/2, 8, 0);
        

    }
}