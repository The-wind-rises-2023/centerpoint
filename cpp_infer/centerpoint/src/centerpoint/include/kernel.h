
#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "params.h"

const int THREADS = 256;    // threads number for a block
const int POINTS_PER_VOXEL = 32;      // depands on "params.h"
const int WARP_SIZE = 32;             // one warp(32 threads) for one pillar
const int WARPS_PER_BLOCK = 4;        // four warp for one block
const int FEATURES_SIZE = 10;         // features maps number depands on "params.h"
const int PILLARS_PER_BLOCK = 64;     // one thread deals with one pillar and a block has PILLARS_PER_BLOCK threads
const int PILLAR_FEATURE_SIZE = 64;   // feature count for one pillar depands on "params.h"

typedef enum
{
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

cudaError_t generateVoxels_random_launch(float *points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size,
        unsigned int *mask, float *voxels,
        cudaStream_t stream = 0);

cudaError_t generateBaseFeatures_launch(unsigned int *mask, float *voxels,
        int grid_y_size, int grid_x_size,
        unsigned int *pillar_num,
        float *voxel_features,
        unsigned int *voxel_num,
        unsigned int *voxel_idxs,
        cudaStream_t stream = 0);

cudaError_t generateFeatures_launch(float* voxel_features,
    unsigned int *voxel_num,
    unsigned int *voxel_idxs,
    unsigned int *params,
    float voxel_x, float voxel_y, float voxel_z,
    float range_min_x, float range_min_y, float range_min_z,
    float* features,
    cudaStream_t stream = 0);

cudaError_t postprocess_launch(const float *bboxes, 
                      const float *scores, 
                      const int *labels, 
                      const float score_thresh,
                      float *bboxes_res, int* bboxes_labels,
                      int *bboxes_num,
                      cudaStream_t stream = 0);

cudaError_t seg_postprocess_launch(const float *semantic, 
                                   int grid_x_size, int grid_y_size,
                                   int num_class, int *seg_res, float *seg_scores,
                                   cudaStream_t stream = 0);
#endif
