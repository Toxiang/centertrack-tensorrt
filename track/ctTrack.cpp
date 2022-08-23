
#include<assert.h>
#include<fstream>
#include"entroyCalibrator.h"
#include"ctTrack.h"
// #include"trackForwardGPU.h"
#include"trackForwardGPU.h"



namespace Track 
{
	//track 构造函数 ：构建引擎
	ctTrack::ctTrack(const std::string& onnxFile,const std::string& calibFile,Track::MODE mode): 
		mContext(nullptr),mEngine(nullptr),mRunTime(nullptr),Mode(mode),runItems(0)
	{
		using namespace std;
		nvinfer1::IHostMemory *modelStream{nullptr};
		int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
	
        memset(&mParams,0,sizeof(mParams));
         mParams.onnxFileName = onnxFile;

        const int maxBatchSize = 1;
    
        auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));

        
        if (!builder)
        {
            std::cout<<"build failed!"<<std::endl;
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

        if (!network)
        {
            std::cout<<"build failed!"<<std::endl;
        }

        auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
        {
            std::cout<<"build failed!"<<std::endl;
        }

        auto parser
            = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        if (!parser)
        {
            std::cout<<"build failed!"<<std::endl;
        }

        auto constructed = constructNetwork(builder, network, config, parser);
        if (!constructed)
        {
            std::cout<<"build failed!"<<std::endl;
        }

        // CUDA stream used for profiling by the builder.
        auto profileStream = samplesCommon::makeCudaStream();
        if (!profileStream)
        {
            std::cout<<"build failed!"<<std::endl;
        }

        config->setProfileStream(*profileStream);

        SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan)
        {
            std::cout<<"build failed!"<<std::endl;
        }

        SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
        if (!runtime)
        {
            std::cout<<"build failed!"<<std::endl;
        }

        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
        if (!mEngine)
        {
            std::cout<<"build failed!"<<std::endl;
        }



        //ASSERT(network->getNbInputs() == 1);
        mInputDims = network->getInput(0)->getDimensions();
       // ASSERT(mInputDims.nbDims == 4);

       // ASSERT(network->getNbOutputs() == 1);
        mOutputDims = network->getOutput(0)->getDimensions();
       // ASSERT(mOutputDims.nbDims == 2);   



		builder->setMaxBatchSize(maxBatchSize);
		//builder->setMaxWorkspaceSize(1<<30);//1G

		// nvinfer1::int8EntroyCalibrator *calibrator = nullptr;
		// if(calibFile.size()>0)
		// 	calibrator = new nvinfer1::int8EntroyCalibrator(maxBatchSize,calibFile,"calib.table");
		cout<<"---------开始构建引擎----------"<<endl;

        // use config to determine which mode ultilized
        if (Mode== MODE::INT8)
        {
            //nvinfer1::IInt8Calibrator* calibrator;
            std::cout <<"setInt8Mode"<<std::endl;
            if (!builder->platformHasFastInt8())
            {
                std::cout << "Notice: the platform do not has fast for int8" << std::endl;
            }  
        }
        else if (Mode == MODE::FLOAT16)
        {
            std::cout <<"setFp16Mode"<<std::endl;
            if (!builder->platformHasFastFp16())
                std::cout << "Notice: the platform do not has fast for fp16" << std::endl;
          
        }
        else if (Mode == MODE::FLOAT32)
        {
            std::cout <<"setFp32Mode"<<std::endl;
            if (!builder->platformHasTf32())
                std::cout << "Notice: the platform do not has fast for fp16" << std::endl;
        
        };
        // config input shape

        //nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);

        nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network,*config);
        if (!engine){
            std::string error_message ="Unable to create engine";
            sample::gLogger.log(nvinfer1::ILogger::Severity::kERROR, error_message.c_str());
            exit(-1);
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
        mRunTime = nvinfer1::createInferRuntime(sample::gLogger);
        assert(mRunTime != nullptr);
       // mEngine= mRunTime->deserializeCudaEngine(modelStream->data(), modelStream->size(), mPlugins);
        assert(mEngine != nullptr);
        modelStream->destroy();
        initEngine();

	}
	//读取引擎文件反序列化 
	ctTrack::ctTrack(const std::string& engineFile):
	mContext(nullptr),mEngine(nullptr),mRunTime(nullptr),Mode(MODE::FLOAT32),runItems(0)
           
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

	
		cout<<"--------------反序列化--------------"<<endl;
		mRunTime = nvinfer1::createInferRuntime(sample::gLogger);
		assert(mRunTime!=nullptr);
		//mEngine = mRunTime->deserializeCudaEngine(data.get(),length,mPlugins);
		assert(mEngine!=nullptr);
		initEngine();
	}

	   bool ctTrack::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{

     std::cout<<"wilson  constructNetwork<<<<<<<<<<<1 \n"<<std::endl;
     printf("mParams.onnxFileName:%s \n",mParams.onnxFileName.c_str());
     printf("mParams.dataDirs:%d \n",mParams.dataDirs.size());
     std::string onnxpath = "model/nuScenes_3Dtracking.onnx";
    //auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        //static_cast<int>(sample::gLogger.getReportableSeverity()));
    auto parsed = parser->parseFromFile(onnxpath.c_str(),static_cast<int>(sample::gLogger.getReportableSeverity()));



        
    printf("parsed = %d \n",parsed);
      std::cout<<"wilson  constructNetwork<<<<<<<<<<<1.1 \n"<<std::endl;
    if (!parsed)
    {
        std::cout<<"parsed is null, return false \n"<<std::endl;
        return false;
    }

    config->setMaxWorkspaceSize(16_MiB);
     std::cout<<"wilson  constructNetwork<<<<<<<<<<<1.2 \n"<<std::endl;
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }
    std::cout<<"wilson  constructNetwork<<<<<<<<<<<1.3 \n"<<std::endl;
    //samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
    std::cout<<"wilson  constructNetwork<<<<<<<<<<<2 \n"<<std::endl;
    return true;
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

		int w = Track::input_w;

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
