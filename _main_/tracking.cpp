// 
// Toxic code in 2021.5.17
// 

#include<argparse.h>
#include<string>
#include<iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include<memory>
#include<typeinfo>
#include"ctTrack.h"
#include"utils.h"

int main(int argc,const char**argv){
    optparse::OptionParser parser;
    parser.add_option("-e","--input-engine-file").dest("engineFile").help("引擎文件");
    parser.add_option("-i","--input-img-file").dest("imgFile").set_default("1.jpg");
    // parser.add_option("-c","input-video-file").dest("capFile").set_default("test.h264");
    optparse::Values options = parser.parse_args(argc,argv);
    if(options["engineFile"].size()==0){
        std::cout<<"-------------没有输入引擎文件-----------------"<<std::endl;
        exit(-1);
    }

    cv::RNG rng(244);
    std::vector<cv::Scalar>color = {cv::Scalar(255,0,0),cv::Scalar(0,255,0)};
    for(int i=0;i<Track::classNum;i++)
        color.push_back(randomColor(rng));

    cv::namedWindow("识别结果",cv::WINDOW_NORMAL);
    
    // cv::Mat ii = cv::imread(options["imgFile"]);
    // cv::resizeWindow("识别结果",320,96);
    // cv::imshow("识别结果",ii);
    
    Track::ctTrack tk(options["engineFile"]);
    std::unique_ptr<float[]> res_img(new float[tk.outputBufferSize]);
    cv::Mat cur_img;
    cv::Mat pre_img;
    if(options["imgFile"].size()>0)
    {
        cur_img = cv::imread(options["imgFile"]);
        pre_img = cv::imread("2.jpg");
        auto pre_data = prepareImage(pre_img);
        auto cur_data = prepareImage(cur_img);
        std::cout<<cur_img.rows<<" "<<cur_img.cols<<std::endl;
        std::vector<float>pre_hm;
        for(int i=0;i<491520;i++){
            pre_hm.push_back(0);
        }
        std::vector<Detection> result;
        //inference1 c++
        // tk.doInference(cur_data.data(),pre_data.data(),pre_hm.data(),result,cur_img);
        // tk.printTime();

        //使用inference2 走.cu
        tk.doInference2(cur_data.data(),pre_data.data(),pre_hm.data(),res_img.get());
        int num_det = static_cast<int>(res_img[0]);
        // std::cout<<"检测到的det:"<<num_det<<std::endl;
        // 
        std::vector<Detection>::iterator iter;
        result.resize(num_det);
        memcpy(result.data(), &res_img[1], num_det * sizeof(Detection));
        for(iter=result.begin();iter!=result.end();iter++){
            std::cout<<"x1:"<<(*iter).bbox.x1<<std::endl;
            std::cout<<"y1:"<<(*iter).bbox.y1<<std::endl;
            std::cout<<"x2:"<<(*iter).bbox.x2<<std::endl;
            std::cout<<"y2:"<<(*iter).bbox.y2<<std::endl;
            std::cout<<"classId:"<<(*iter).classId<<std::endl;
            std::cout<<"prob:"<<(*iter).prob<<std::endl;
        }
        
        postProcess(result,cur_img);
        drawImg(result,cur_img,color);
	    cv::resizeWindow("识别结果",1024,1024);
        cv::imshow("识别结果",cur_img);
        cv::waitKey(0);
    }
    // if(options["capFile"].size()>0)
    // {
        
    //     std::cout<<"视频跟踪！"<<std::endl;
    //     cv::VideoCapture cap(options["capFile"]);
    //     // CvCapture * cc = cvCaptureFromFile("1.mp4");
    //     // int frameH    = (int) cvGetCaptureProperty(cc, CV_CAP_PROP_FRAME_HEIGHT);  
    //     // int frameW    = (int) cvGetCaptureProperty(cc, CV_CAP_PROP_FRAME_WIDTH);  
    //     // int fps       = (int) cvGetCaptureProperty(cc, CV_CAP_PROP_FPS); 
    //     // std::cout<<frameH<<" "<<frameW<<s" "<<fps<<std::endl;
    //     std::cout<<cap.get(CV_CAP_PROP_FRAME_HEIGHT)<<" "<<cap.get(CV_CAP_PROP_FRAME_WIDTH)<<cap.get(CV_CAP_PROP_FPS)<<std::endl;
    //     cv::Size size = cv::Size(1920,1088);
    //     cv::VideoWriter out("res-3.avi",CV_FOURCC('D', 'I', 'V', 'X'),80,size);
    //     while (cap.read(cur_img))
    //     {
            
    //         std::vector<float>pre_hm;
    //         for(int i=0;i<491520;i++){
    //             pre_hm.push_back(0);
    //         }
    //         auto inputData = prepareImage(cur_img);
    //         auto preData = prepareImage(cur_img);
    //         tk.doInference(preData.data(),inputData.data(),pre_hm.data(), res_img.get());
    //         // tk.printTime();

    //         // int num_det = static_cast<int>(res_img[0]);

    //         // std::vector<Detection> result;

    //         // result.resize(num_det);

    //         // memcpy(result.data(), &res_img[1], num_det * sizeof(Detection));

    //         // postProcess(result,cur_img);

    //         // drawImg(result,cur_img,color);
    //         // out.write(cur_img);
    //         // cv::imshow("识别结果",cur_img);
    //         // if((cv::waitKey(1)& 0xff) == 27){
    //         //     cv::destroyAllWindows();
    //         //     return 0;
    //         // };

    //     }

    // }

    std::cout<<"结束！"<<std::endl;
    return 0;



}