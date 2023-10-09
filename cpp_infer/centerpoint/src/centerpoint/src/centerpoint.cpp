#include "centerpoint.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"

TRT::~TRT(void)
{
  checkCudaErrors(cudaEventDestroy(start_));
  checkCudaErrors(cudaEventDestroy(stop_));
  return;
}

TRT::TRT(std::string modelFile, cudaStream_t stream):stream_(stream)
{
  //std::string modelCache = modelFile + ".cache";
  std::string modelCache = modelFile + ".plan";
  std::fstream trtCache(modelCache, std::ifstream::in);
  checkCudaErrors(cudaEventCreate(&start_));
  checkCudaErrors(cudaEventCreate(&stop_));
  initLibNvInferPlugins(&gLogger_, "");
  std::string plugin_path = "src/centerpoint/3rd_party/libmmdeploy_tensorrt_ops.so";
  void* plugin_handle = dlopen(plugin_path.c_str(), RTLD_LAZY);
  if (plugin_handle == nullptr) {
    std::cout << "load plugin failed, errors: " << dlerror() << std::endl;
    exit(-1);
  } else {
    std::cout << "load plugin success" << std::endl;
  }
  
  if (!trtCache.is_open())
  {
    std::cerr << "Error:" << modelCache << " not found!" << std::endl;
    exit(-1);
   
  } else {
	  std::cout << "load TRT cache."<<std::endl;
    char *data;
    unsigned int length;

    // get length of file:
    trtCache.seekg(0, trtCache.end);
    length = trtCache.tellg();
    trtCache.seekg(0, trtCache.beg);

    data = (char *)malloc(length);
    if (data == NULL ) {
       std::cout << "Can't malloc data.\n";
       exit(-1);
    }

    trtCache.read(data, length);
    // create context
    auto runtime = nvinfer1::createInferRuntime(gLogger_);

    if (runtime == nullptr) {	  std::cout << "load TRT cache0."<<std::endl;
        std::cerr << ": runtime null!" << std::endl;
        exit(-1);
    }
    //plugin_ = nvonnxparser::createPluginFactory(gLogger_);
    engine_ = (runtime->deserializeCudaEngine(data, length, 0));
    if (engine_ == nullptr) {
        std::cerr << ": engine null!" << std::endl;
        exit(-1);
    }
    free(data);


    trtCache.close();
  }

  context_ = engine_->createExecutionContext();

  // for dynamic input shape,by xing
  context_->setOptimizationProfileAsync(0, stream_);

  return;
}

int TRT::doinfer(void**buffers)
{
  int status;

  status = context_->enqueueV2(buffers, stream_, &start_);

  if (!status)
  {
      return -1;
  }

  return 0;
}

PointPillar::PointPillar(std::string modelFile, cudaStream_t stream):stream_(stream)
{
  checkCudaErrors(cudaEventCreate(&start_));
  checkCudaErrors(cudaEventCreate(&stop_));

  pre_.reset(new PreProcessCuda(stream_));
  trt_.reset(new TRT(modelFile, stream_));
  post_.reset(new PostProcessCuda(stream_));

  //point cloud to voxels
  voxel_features_size_ = MAX_VOXELS * params_.max_num_points_per_pillar * 4 * sizeof(float);
  voxel_num_size_ = MAX_VOXELS * sizeof(unsigned int);
  voxel_idxs_size_ = MAX_VOXELS* 4 * sizeof(unsigned int);

  checkCudaErrors(cudaMallocManaged((void **)&voxel_features_, voxel_features_size_));
  checkCudaErrors(cudaMallocManaged((void **)&voxel_num_, voxel_num_size_));
  checkCudaErrors(cudaMallocManaged((void **)&voxel_idxs_, voxel_idxs_size_));

  checkCudaErrors(cudaMemsetAsync(voxel_features_, 0, voxel_features_size_, stream_));
  checkCudaErrors(cudaMemsetAsync(voxel_num_, 0, voxel_num_size_, stream_));
  checkCudaErrors(cudaMemsetAsync(voxel_idxs_, 0, voxel_idxs_size_, stream_));

  //TRT-input
  checkCudaErrors(cudaMallocManaged((void **)&params_input_, sizeof(unsigned int)));

  // checkCudaErrors(cudaMemsetAsync(features_input_, 0, features_input_size_, stream_));
  checkCudaErrors(cudaMemsetAsync(params_input_, 0, sizeof(unsigned int), stream_));

  //output of TRT -- input of post-process
  bboxes_size_ = params_.bboxes_dim1 * params_.booxes_dim2 * params_.booxes_dim3 * sizeof(float);
  scores_size_ = params_.scores_dim1 * params_.scores_dim2 * sizeof(float);
  labels_size_ = params_.labels_dim1 * params_.labels_dim2 * sizeof(int);
  semantic_size_ = params_.seg_num_classes * params_.grid_x_size * params_.grid_y_size * sizeof(float);

  checkCudaErrors(cudaMallocManaged((void **)&bboxes_output_, bboxes_size_));
  checkCudaErrors(cudaMallocManaged((void **)&scores_output_, scores_size_));
  checkCudaErrors(cudaMallocManaged((void **)&labels_output_, labels_size_));
  checkCudaErrors(cudaMallocManaged((void **)&semantic_output_, semantic_size_));

  //output of post-process
  // todo: add semantic output

  res_.reserve(100);
  return;
}

PointPillar::~PointPillar(void)
{
  pre_.reset();
  trt_.reset();
  post_.reset();

  checkCudaErrors(cudaFree(voxel_features_));
  checkCudaErrors(cudaFree(voxel_num_));
  checkCudaErrors(cudaFree(voxel_idxs_));

  // checkCudaErrors(cudaFree(features_input_));
  checkCudaErrors(cudaFree(params_input_));

  checkCudaErrors(cudaFree(bboxes_output_));
  checkCudaErrors(cudaFree(scores_output_));
  checkCudaErrors(cudaFree(labels_output_));
  checkCudaErrors(cudaFree(semantic_output_));

  checkCudaErrors(cudaEventDestroy(start_));
  checkCudaErrors(cudaEventDestroy(stop_));
  return;
}

int PointPillar::doinfer(void*points_data, unsigned int points_size, std::vector<Bndbox> &nms_pred)
{
#if PERFORMANCE_LOG
  float generateVoxelsTime = 0.0f;
  checkCudaErrors(cudaEventRecord(start_, stream_));
#endif
  checkCudaErrors(cudaMemsetAsync(voxel_features_, 0, voxel_features_size_, stream_));
  checkCudaErrors(cudaMemsetAsync(voxel_num_, 0, voxel_num_size_, stream_));
  checkCudaErrors(cudaMemsetAsync(voxel_idxs_, 0, voxel_idxs_size_, stream_));
  checkCudaErrors(cudaDeviceSynchronize());
  pre_->generateVoxels((float*)points_data, points_size,
        params_input_,
        voxel_features_, 
        voxel_num_,
        voxel_idxs_);

#if PERFORMANCE_LOG
  checkCudaErrors(cudaEventRecord(stop_, stream_));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventElapsedTime(&generateVoxelsTime, start_, stop_));
  unsigned int params_input_cpu;
  checkCudaErrors(cudaMemcpy(&params_input_cpu, params_input_, sizeof(unsigned int), cudaMemcpyDefault));
  std::cout<<"find pillar_num: "<< params_input_cpu <<std::endl;
#endif

#if PERFORMANCE_LOG
  float doinferTime = 0.0f;
  checkCudaErrors(cudaEventRecord(start_, stream_));
#endif

  // for dynamic shape, set bindings , by xing
  unsigned int voxel_num;
  
  checkCudaErrors(cudaMemcpy(&voxel_num, params_input_, sizeof(unsigned int), cudaMemcpyDefault));
  trt_->context()->setBindingDimensions(0, nvinfer1::Dims3(voxel_num, params_.max_num_points_per_pillar, 4)); 
  nvinfer1::Dims num_dim;
  num_dim.nbDims = 1;
  num_dim.d[0] = voxel_num;
  trt_->context()->setBindingDimensions(1, num_dim);
  trt_->context()->setBindingDimensions(2, nvinfer1::Dims2(voxel_num, 4)); 

  void *buffers[] = {voxel_features_, voxel_num_, voxel_idxs_, bboxes_output_, scores_output_, labels_output_, semantic_output_};
  trt_->doinfer(buffers);
  
  checkCudaErrors(cudaMemsetAsync(params_input_, 0, sizeof(unsigned int), stream_));
  checkCudaErrors(cudaDeviceSynchronize());
#if PERFORMANCE_LOG
  checkCudaErrors(cudaEventRecord(stop_, stream_));
  checkCudaErrors(cudaEventSynchronize(stop_));
  checkCudaErrors(cudaEventElapsedTime(&doinferTime, start_, stop_));
#endif

#if PERFORMANCE_LOG
  float doPostprocessCudaTime = 0.0f;
  checkCudaErrors(cudaEventRecord(start_, stream_));
#endif

  post_->doPostprocessCuda( bboxes_output_, scores_output_, labels_output_, semantic_output_, res_, seg_bev_);
  
  checkCudaErrors(cudaDeviceSynchronize());

  nms_cpu(res_, params_.nms_thresh, nms_pred);
  std::cout << "res.size: " << res_.size() << "nms_pred size: " << nms_pred.size() << std::endl;

  res_.clear();

#if PERFORMANCE_LOG
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(stop_, stream_));
  checkCudaErrors(cudaEventSynchronize(stop_));
  checkCudaErrors(cudaEventElapsedTime(&doPostprocessCudaTime, start_, stop_));
  std::cout<<"TIME: generateVoxels: "<< generateVoxelsTime <<" ms." <<std::endl;
  std::cout<<"TIME: doinfer: "<< doinferTime <<" ms." <<std::endl;
  std::cout<<"TIME: doPostprocessCuda: "<< doPostprocessCudaTime <<" ms." <<std::endl;
#endif
  return 0;
}
