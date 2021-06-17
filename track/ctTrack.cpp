//
// Toxic code in 2021.5.17
//

#include<assert.h>
#include<fstream>
#include"entroyCalibrator.h"
#include"ctTrack.h"
// #include"trackForwardGPU.h"
#include"trackForwardGPU.h"

static Logger gLogger;

namespace Track 
{
	//track 构造函数 ：构建引擎
	ctTrack::ctTrack(const std::string& onnxFile,const std::string& calibFile,Track::MODE mode): 
		mContext(nullptr),mEngine(nullptr),mRunTime(nullptr),Mode(mode),runItems(0),mPlugins(nullptr)
	{
		using namespace std;
		nvinfer1::IHostMemory *modelStream{nullptr};
		int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
		nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
		nvinfer1::INetworkDefinition* network = builder->createNetwork();
		
		mPlugins = nvonnxparser::createPluginFactory(gLogger);
		auto parser = nvonnxparser::createParser(*network,gLogger);
		if(!parser->parseFromFile(onnxFile.c_str(),verbosity))
		{
			string msg("----------解析onnx文件失败-----------");
			gLogger.log(nvinfer1::ILogger::Severity::kERROR,msg.c_str());
			exit(EXIT_FAILURE);
		}

		const int maxBatchSize = 1;
		builder->setMaxBatchSize(maxBatchSize);
		builder->setMaxWorkspaceSize(1<<30);//1G

		// nvinfer1::int8EntroyCalibrator *calibrator = nullptr;
		// if(calibFile.size()>0)
		// 	calibrator = new nvinfer1::int8EntroyCalibrator(maxBatchSize,calibFile,"calib.table");
		cout<<"---------开始构建引擎----------"<<endl;
		nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
		if(!engine){
			string error_msg = "无法创建引擎";
			gLogger.log(nvinfer1::ILogger::Severity::kERROR,error_msg.c_str());
			exit(EXIT_FAILURE);
		}
		cout<<"-------------引擎构建结束---------------"<<endl;

		// if(calibrator){
		// 	delete calibrator;
		// 	calibrator = nullptr;
		// }

		// 序列化引擎

		modelStream = engine->serialize();
		cout<<"ssss"<<std::endl;
		engine->destroy();
        network->destroy();
        builder->destroy();
        parser->destroy();
        assert(modelStream != nullptr);
        mRunTime = nvinfer1::createInferRuntime(gLogger);
        assert(mRunTime != nullptr);
        mEngine= mRunTime->deserializeCudaEngine(modelStream->data(), modelStream->size(), mPlugins);
        assert(mEngine != nullptr);
        modelStream->destroy();
        initEngine();

	}
	//读取引擎文件反序列化 
	ctTrack::ctTrack(const std::string& engineFile):
	mContext(nullptr),mEngine(nullptr),mRunTime(nullptr),Mode(MODE::FLOAT32),runItems(0),
            mPlugins(nullptr)
	{
		using namespace std;

		fstream file;
		file.open(engineFile,ios::binary|ios::in);
		if(!file.is_open())
		{
			cout<<"------------读取引擎文件失败-------------"<<endl;
			return;
		}
		file.seekg(0,ios::end);
		int length = file.tellg();
		file.seekg(0,ios::beg);
		unique_ptr<char[]>data(new char[length]);
		file.read(data.get(),length);
		file.close();

		mPlugins = nvonnxparser::createPluginFactory(gLogger);
		cout<<"--------------反序列化--------------"<<endl;
		mRunTime = nvinfer1::createInferRuntime(gLogger);
		assert(mRunTime!=nullptr);
		mEngine = mRunTime->deserializeCudaEngine(data.get(),length,mPlugins);
		assert(mEngine!=nullptr);
		initEngine();
	}
	//初始化引擎
	void ctTrack::initEngine()
	{
		// std::cout<<"iii"<<std::endl;
		mContext = mEngine->createExecutionContext();
		// std::cout<<"qqq"<<std::endl;
		assert(mContext!=nullptr);
		// std::cout<<"rrr"<<std::endl;
		mContext->setProfiler(&mProfiler);
		// std::cout<<"ppp"<<std::endl;
		int nbBindings = mEngine->getNbBindings();
		if(nbBindings!=7) std::cout<<"------------模型参数错误-------------"<<std::endl;

		
		mBindBufferSizes.resize(nbBindings);
		int64_t totalsize = 0;
		const int maxBatchSize = 1;
		// std::cout<<"ooo"<<std::endl;
		for(int i=0;i<nbBindings;i++){
			nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
			// std::cout<<"1"<<std::endl;
			nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
			// std::cout<<"2"<<std::endl;
			// 注意const int 和 int相乘 会报错
			std::cout<<"binding"<<i+1<<":"<<volume(dims)<<std::endl;
			std::cout<<getElementSize(dtype)<<std::endl;
			totalsize = volume(dims)*maxBatchSize*getElementSize(dtype);
			// std::cout<<"3"<<std::endl;
			mBindBufferSizes[i] = totalsize;
			// std::cout<<"4"<<std::endl;
			mCudaBuffers[i] = safeCudaMalloc(totalsize);
		}
		// std::cout<<"lll"<<std::endl;
		//cudaStream_t mCudaStream // 创建CUDA流以执行此推断
		
		outSize1 = mBindBufferSizes[3]/sizeof(float);
		outSize2 = mBindBufferSizes[4]/sizeof(float);
		outSize3 = mBindBufferSizes[5]/sizeof(float);
		outSize4 = mBindBufferSizes[6]/sizeof(float);
		
		out1 = new float[outSize1];
		out2 = new float[outSize2];
		out3 = new float[outSize3];
		out4 = new float[outSize4];
		
        outputBufferSize = mBindBufferSizes[3]*10;
        cudaOutputBuffer = safeCudaMalloc(outputBufferSize);
		CUDA_CHECK(cudaStreamCreate(&mCudaStream));
	}
	
	//前向推理 处理部分使用cuda加速
	void ctTrack::doInference2(const void *cur_img,const void *pre_img,const void *pre_hm,
							  void* res_img){
		// 将数据从主机输入缓冲区异步复制到设备输入缓冲区
		CUDA_CHECK(cudaMemcpyAsync(mCudaBuffers[0],cur_img,mBindBufferSizes[0],cudaMemcpyHostToDevice, mCudaStream));
		CUDA_CHECK(cudaMemcpyAsync(mCudaBuffers[1],pre_img,mBindBufferSizes[1],cudaMemcpyHostToDevice, mCudaStream));
		CUDA_CHECK(cudaMemcpyAsync(mCudaBuffers[2],pre_hm, mBindBufferSizes[2],cudaMemcpyHostToDevice, mCudaStream));
		const int batchSize = 1;
		mContext->execute(batchSize, mCudaBuffers);
		CUDA_CHECK(cudaMemset(cudaOutputBuffer, 0, sizeof(float)));
		
		ctTrack_Forward_GPU(static_cast<const float *>(mCudaBuffers[3]),static_cast<const float *>(mCudaBuffers[4]),
							static_cast<const float *>(mCudaBuffers[6]),static_cast<float *>(cudaOutputBuffer),
							input_w/4,input_h/4,classNum,kernelSize,visThresh);
		
	/*
		int h=96,w=320;
		// 
		cv::Mat m;
		cv::Mat mm;
		cv::cvtColor(img,m,CV_BGR2GRAY);
		cv::resize(m,mm,cv::Size(320,96),0,0,CV_INTER_LINEAR);
		cv::Mat m1 = cv::Mat(96,320,CV_8UC1);
		cv::Mat m2 = cv::Mat(96,320,CV_8UC1);
		cv::Mat m3 = cv::Mat(96,320,CV_8UC1);
		
		cv::imshow("ori_pic",mm);
		cv::waitKey(0);
		auto sigmoid = [&](float x){return (1 / (1 + exp(-x)));};
		for(int i=0;i<30720;i++){
			float prob = sigmoid(out1[i]);
			int row = i/w%h;
			int col = i%w;
			m1.at<uchar>(row,col) = int(prob*255*3);
			if(prob>0.4)
				mm.at<uchar>(row,col) = int(prob*255);
		}
		for(int i=30720;i<61440;i++){
			float prob = sigmoid(out1[i]);
			int row = i/w%h;
			int col = i%w;
			m2.at<uchar>(row,col) = int(prob*255*3);
			if(prob>0.4)
				mm.at<uchar>(row,col) = int(prob*255);
		}
		for(int i=61440;i<92160;i++){
			float prob = sigmoid(out1[i]);
			int row = i/w%h;
			int col = i%w;
			m3.at<uchar>(row,col) = int(prob*255*3);
			if(prob>0.4)
				mm.at<uchar>(row,col) = int(prob*255);
		}
		cv::namedWindow("pic",cv::WINDOW_NORMAL);
		cv::resizeWindow("pic",1024,1024);
		cv::imshow("pic",mm);
		
		cv::waitKey(0);
		cv::namedWindow("hm1",cv::WINDOW_NORMAL);
		cv::resizeWindow("hm1",1024,1024);
		cv::imshow("hm1",m1);
		cv::waitKey(0);
		cv::namedWindow("hm2",cv::WINDOW_NORMAL);
		// cv::resizeWindow("hm2",320,96);
		cv::imshow("hm2",m2);
		cv::waitKey(0);
		cv::namedWindow("hm3",cv::WINDOW_NORMAL);
		// cv::resizeWindow("hm3",320,96);
		cv::imshow("hm3",m3);
		cv::waitKey(0);
		*/
		
		CUDA_CHECK(cudaMemcpyAsync(res_img,cudaOutputBuffer,outputBufferSize,cudaMemcpyDeviceToHost,mCudaStream));
		runItems++;
	}
	// 正常推理 结果处理未使用cuda
	void ctTrack::doInference(const void *cur_img,const void *pre_img,const void *pre_hm,
							  std::vector<Detection>& res_img,cv::Mat& img){
		// 将数据从主机输入缓冲区异步复制到设备输入缓冲区
		CUDA_CHECK(cudaMemcpyAsync(mCudaBuffers[0],cur_img,mBindBufferSizes[0],cudaMemcpyHostToDevice, mCudaStream));
		CUDA_CHECK(cudaMemcpyAsync(mCudaBuffers[1],pre_img,mBindBufferSizes[1],cudaMemcpyHostToDevice, mCudaStream));
		CUDA_CHECK(cudaMemcpyAsync(mCudaBuffers[2],pre_hm, mBindBufferSizes[2],cudaMemcpyHostToDevice, mCudaStream));
		const int batchSize = 1;
		mContext->execute(batchSize, mCudaBuffers);
		CUDA_CHECK(cudaMemset(cudaOutputBuffer, 0, sizeof(float)));
		
		
		CUDA_CHECK(cudaMemcpyAsync(out1,mCudaBuffers[3],mBindBufferSizes[3],cudaMemcpyDeviceToHost,mCudaStream));
		CUDA_CHECK(cudaMemcpyAsync(out2,mCudaBuffers[4],mBindBufferSizes[4],cudaMemcpyDeviceToHost,mCudaStream));
		CUDA_CHECK(cudaMemcpyAsync(out3,mCudaBuffers[5],mBindBufferSizes[5],cudaMemcpyDeviceToHost,mCudaStream));
		CUDA_CHECK(cudaMemcpyAsync(out4,mCudaBuffers[6],mBindBufferSizes[6],cudaMemcpyDeviceToHost,mCudaStream));

		
		std::vector<Detection> res_imgs;
		res_imgs = processDet();
		res_img = res_imgs;

		// int num_det = static_cast<int>(res_imgs[0]);
        // std::cout<<"检测到的det:"<<num_det<<std::endl;
        // std::vector<Detection> result;
        std::vector<Detection>::iterator iter;
		int num=1;
        for(iter=res_imgs.begin();iter!=res_imgs.end();iter++){
			std::cout<<"num:"<<num<<std::endl;
            std::cout<<"prob:"<<(*iter).prob<<std::endl;
			num++;
        }
		
	}
	//

	void ctTrack::saveEngine(const std::string& fileName)
	{
		if(mEngine)
		{
			nvinfer1::IHostMemory* data = mEngine->serialize();
            std::ofstream file;
            file.open(fileName,std::ios::binary | std::ios::out);
            if(!file.is_open())
            {
                std::cout << "创建引擎文件" << fileName <<"失败" << std::endl;
                return;
            }
            file.write((const char*)data->data(), data->size());
            file.close();
		}
	}

	//未加后处理
	std::vector<Detection> ctTrack::processDet(){
		
		std::cout<<"开始处理det"<<std::endl;
		
		int y, x;
		int w = 320;
		int h = 96;
		// 取前100
		int topK = 100;
		int classNum = 3; 
		int padding = 1;
		int offset = -padding;
		// 单通道大小
		int stride = w * h;
		// det
		std::vector<Detection> dets;
		

		auto do_sigmoid = [&](float x){return (1 / (1 + exp(-x)));};
		//对hm做sigmoid 并判断相对于该中心点周围八个点的最大值
		for(int c = 0; c < classNum; c++)
		{
			for (int i = 0; i < h*w; ++i)
			{
				x = i % w;
				y = i / w % h;
				
				// std::cout<<"("<<x<<","<<y<<")"<<std::endl;
				float pprob = 0.0;
				float max = -1;
				int maxIndex = 0;
				// int max_idx = 0;
				// 在hm的九个中心点中遍历
				for(int l=0;l<3;l++)
					for(int m=0;m<3;m++){
						int xx = offset+l+x;
						int curX = (xx>0)?xx:0;
						curX = (curX<w)?curX:w-1;
						int yy = offset+m+y;
						int curY = (yy>0)?yy:0;
						curY = (curY<h)?curY:h-1;
						int curIndex = curY*w+curX+stride*c;
						// int valid = 1;
						// std::cout<<valid<<std::endl;
						float val = do_sigmoid(out1[curIndex]);

						maxIndex = (val>max)?curIndex:maxIndex;
						max = (val>max)?val:max;
						// std::cout<<"max_index:"<<max_index<<" max:"<<max<<std::endl;
					}

				if(maxIndex == c * w * h + i){
					// std::cout<<out1[c * w * h + i]<<" ";
					pprob = do_sigmoid(out1[maxIndex]);

				}
				if(pprob>0.4){
					std::cout<<pprob<<" ";
					Detection d;
					//reg 两通道 1：i 2：i+stride
					int X = x+out2[i];
					int Y = y+out2[i+stride];
					d.bbox.x1 = (X-out4[i]/2)*4;
					d.bbox.y1 = (Y-out4[i+stride]/2)*4;
					d.bbox.x2 = (X+out4[i]/2)*4;
					d.bbox.y2 = (Y+out4[i+stride]/2)*4;
					d.classId = c;
					d.prob = pprob;
					d.tracking_x = out3[i];
					d.tracking_y = out3[i+stride];
					dets.push_back(d);
				}
				
			}
		}
		return dets;

	}

}
