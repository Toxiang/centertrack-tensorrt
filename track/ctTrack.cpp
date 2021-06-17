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
    
	//加后处理函数
	std::vector<Detection> ctTrack::postProcessDet(){
		/* Synchronize different cuda stream */
    	cudaStreamSynchronize(mCudaStream);
		float hm_dat[96*320*3];
		float reg_data[96*320*2];
		float wh_data[96*320*2];
		float tracking_data[96*320*2];
		std::cout<<"开始处理det"<<std::endl;
		
		CUDA_CHECK(cudaMemcpyAsync((void*)hm_dat, mCudaBuffers[3], mBindBufferSizes[3], cudaMemcpyDeviceToHost, mCudaStream));
		CUDA_CHECK(cudaMemcpyAsync((void*)reg_data, mCudaBuffers[4], mBindBufferSizes[4], cudaMemcpyDeviceToHost, mCudaStream));
		CUDA_CHECK(cudaMemcpyAsync((void*)wh_data, mCudaBuffers[6], mBindBufferSizes[6], cudaMemcpyDeviceToHost, mCudaStream));
		CUDA_CHECK(cudaMemcpyAsync((void*)tracking_data, mCudaBuffers[5], mBindBufferSizes[5], cudaMemcpyDeviceToHost, mCudaStream));

		
		cudaStreamSynchronize(mCudaStream);
		int grid_y, grid_x;
		int hm_w        = 320;
		int hm_h        = 96;
		int ori_w       = 1600;//cols
		int ori_h       = 1017;//rows
		int topK        = 100;
		int num_class   = 3; 
		int padding = (3-1)/2;
		int offset = -padding;
		int stride = hm_w * hm_h;
		// std::vector<Detection> dets;
		
		// int filter_w    = 3;
		//初始化hm_data 
		float *hm_data = new float[num_class * hm_w * hm_h] ;
		for(int kk=0;kk<num_class * hm_w * hm_h;kk++){
			hm_data[kk] = 0;
		}

		auto do_sigmoid = [&](float x){return (1 / (1 + exp(-x)));};
		//对hm做sigmoid
		for(int c = 0; c < 3; c++)
		{
			for (int i = 0; i < hm_h*hm_w; ++i)
			{
				grid_y = i / hm_w % hm_h;
				grid_x = i % hm_w;
				// std::cout<<"("<<sgrid_x<<","<<grid_y<<")"<<std::endl;
				int cls = c;
				int reg_index = i;
				float max_val = -1;
				float max = -1;
				int max_index = 0;
				int max_idx = 0;
				// 在hm的九个中心点中遍历
				for(int l=0;l<3;l++)
					for(int m=0;m<3;m++){
						int xx = offset+l+grid_x;
						int cur_x = (xx>0)?xx:0;
						cur_x = (cur_x<hm_w)?cur_x:hm_w-1;
						int yy = offset+m+grid_y;
						int cur_y = (yy>0)?yy:0;
						cur_y = (cur_y<hm_h)?cur_y:hm_h-1;
						int cur_index = cur_y*hm_w+cur_x+stride*cls;
						int valid = 1;
						// std::cout<<valid<<std::endl;
						float val = do_sigmoid(hm_dat[cur_index]);

						max_index = (val>max)?cur_index:max_index;
						max = (val>max)?val:max;
						// std::cout<<"max_index:"<<max_index<<" max:"<<max<<std::endl;
					}
				// // std::cout<<"************************************"<<std::endl;
				// for(int filter_r = -filter_w/2; filter_r <= filter_w/2; ++filter_r)
				// 	for(int filter_c = -filter_w/2; filter_c <= filter_w/2; ++filter_c)
				// 	{
				// 		int cur_y = std::min(std::max(grid_y + filter_r,0),hm_h-1);
				// 		int cur_x = std::min(std::max(grid_x + filter_c,0),hm_w-1);
				// 		int cur_idx = c * hm_w * hm_h + cur_y * hm_w + cur_x;
				// 		// std::cout<<"("<<cur_x<<","<<cur_y<<")"<<" cur_idx:"<<cur_idx<<std::endl;
				// 		float image_val = do_sigmoid(hm_dat[cur_idx]);
				// 		max_idx = (image_val>max_val)?cur_idx:max_idx;
				// 		max_val = std::max(image_val,max_val);
				// 		// std::cout<<"max_idx:"<<max_idx<<" max_val:"<<max_val<<std::endl;
				// 	}
				// // std::cout<<max_index<<" "<<max_idx<<std::endl;
				if(max_index == c * hm_w * hm_h + i){
					// std::cout<<hm_dat[c * hm_w * hm_h + i]<<" ";
					hm_data[c * hm_w * hm_h + i] = do_sigmoid(hm_dat[c * hm_w * hm_h + i]);
				}
				// if(hm_data[c * hm_w * hm_h+i]>0.4){
				// 	Detection d;
				// 	//reg 两通道 1：i 2：i+stride
				// 	int c_x = grid_x+reg_data[i];
				// 	int c_y = grid_y+reg_data[i+stride];
				// 	d.bbox.x1 = (c_x-wh_data[i]/2)*4;
				// 	d.bbox.y1 = (c_y-wh_data[i+stride]/2)*4;
				// 	d.bbox.x2 = (c_x+wh_data[i]/2)*4;
				// 	d.bbox.y2 = (c_y+wh_data[i+stride]/2)*4;
				// 	d.classId = cls;
				// 	d.prob = hm_data[c * hm_w * hm_h+i];
				// 	dets.push_back(d);
				// }
				
			}
		}
		// return dets;

		const std::vector<float> hm_vec(hm_data, hm_data + num_class * hm_w * hm_h);
		// for(int i=0;i<hm_vec.size();i++) std::cout<<hm_vec[i]<<" ";
		// std::cout<<hm_vec.size()<<std::endl;
		std::vector<int> max_index = sort_index(hm_vec);
		// for(int i=0;i<10;i++) std::cout<<max_index[i]<<" "<<hm_vec[max_index[i]]<<std::endl;

		//后处理
		int channel_size = hm_w * hm_h;
		float scale = std::max(ori_w / float(hm_w), ori_h / float(hm_h));
		float delta_w = (scale * float(hm_w) - ori_w) / 2.;
		float delta_h = (scale * float(hm_h) - ori_h) / 2.;
		std::vector<float> topK_score;
		std::vector<float> topK_x, topK_y;
		std::vector<int> topK_class;
		std::vector<float> reg_x, reg_y;
		std::vector<float> wh_x, wh_y;
		std::vector<float> tracking_x, tracking_y;
		std::vector<Detection> results;
		for (int obj_count = 0; obj_count < topK; ++obj_count)
		{
			int idx = max_index[obj_count];
			if (hm_vec[idx] > 0.3){ /* */
				//    break;
				topK_score.push_back(hm_vec[idx]);
				topK_class.push_back(int(idx / channel_size));
				idx %= channel_size;
				topK_y.push_back(idx / hm_w * scale - delta_h);
				topK_x.push_back(idx % hm_w * scale - delta_w);
				reg_x.push_back(reg_data[idx] * scale);
				reg_y.push_back(reg_data[idx + channel_size] * scale);
				wh_x.push_back(wh_data[idx] * scale);
				wh_y.push_back(wh_data[idx + channel_size] * scale);
				tracking_x.push_back(tracking_data[idx] * scale);
				tracking_y.push_back(tracking_data[idx + channel_size] * scale);
				
				Detection ans;
				
				ans.prob        = topK_score[obj_count];
				ans.classId     = topK_class[obj_count];
				ans.bbox.x1     = topK_x[obj_count] + reg_x[obj_count] - wh_x[obj_count] / 2.;
				ans.bbox.y1     = topK_y[obj_count] + reg_y[obj_count] - wh_y[obj_count] / 2.;
				ans.bbox.x2     = topK_x[obj_count] + reg_x[obj_count] - wh_x[obj_count] / 2. + wh_x[obj_count];
				ans.bbox.y2     = topK_y[obj_count] + reg_y[obj_count] - wh_y[obj_count] / 2. + wh_y[obj_count];
				ans.tracking_x  = tracking_x[obj_count];
				ans.tracking_y  = tracking_y[obj_count];
				
				results.push_back(ans);
			}
		}
		delete[] hm_data;
		return results;

	}

	std::vector<int> ctTrack::sort_index(const std::vector<float> &v)
	{
		std::vector<int> idx(v.size());
		std::iota(idx.begin(), idx.end(), 0);
		std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v[i1] > v[i2]; });
		return idx;
	}

}
