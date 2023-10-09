
#include <memory>
#include "dlfcn.h"
#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"
#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"
#include "postprocess.h"
#include "preprocess.h"

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level message
        //if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR || severity == Severity::kINFO ) {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "trt_infer: " << msg << std::endl;
        }
    }
};

class TRT {
  private:
    Params params_;

    cudaEvent_t start_, stop_;

    Logger gLogger_;
    nvinfer1::IExecutionContext *context_ = nullptr;
    nvinfer1::ICudaEngine *engine_ = nullptr;

    cudaStream_t stream_ = 0;
  public:
    TRT(std::string modelFile, cudaStream_t stream = 0);
    ~TRT(void);

    nvinfer1::ICudaEngine *engine() { return engine_; }
    nvinfer1::IExecutionContext *context() { return context_; }

    int doinfer(void**buffers);
};

class PointPillar {
  private:
    Params params_;

    cudaEvent_t start_, stop_;
    cudaStream_t stream_;

    std::shared_ptr<PreProcessCuda> pre_;
    std::shared_ptr<TRT> trt_;
    std::shared_ptr<PostProcessCuda> post_;

    //input of pre-process
    float *voxel_features_ = nullptr;
    unsigned int *voxel_num_ = nullptr;
    unsigned int *voxel_idxs_ = nullptr;
    unsigned int *pillar_num_ = nullptr;

    unsigned int voxel_features_size_ = 0;
    unsigned int voxel_num_size_ = 0;
    unsigned int voxel_idxs_size_ = 0;

    //TRT-input
    // float *features_input_ = nullptr;
    unsigned int *params_input_ = nullptr;
    // unsigned int features_input_size_ = 0;

    //output of TRT -- input of post-process
    float *bboxes_output_ = nullptr;
    float *scores_output_ = nullptr;
    int *labels_output_ = nullptr;
    float *semantic_output_ = nullptr;
    unsigned int bboxes_size_;
    unsigned int scores_size_;
    unsigned int labels_size_;
    unsigned int semantic_size_;

    std::vector<Bndbox> res_;
    SemanticBev seg_bev_;

  public:
    PointPillar(std::string modelFile, cudaStream_t stream = 0);
    ~PointPillar(void);
    int doinfer(void*points, unsigned int point_size, std::vector<Bndbox> &res);
};
