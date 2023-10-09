
#include "kernel.h"

__global__ void postprocess_kernal(const float *bboxes, 
                      const float *scores, 
                      const int *labels, 
                      const float score_thresh,
                      float *bboxes_res, int* bboxes_labels,
                      int *bboxes_num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1000) {
        return;
    }
    if (scores[i] > score_thresh) {
        int resCount = (int)atomicAdd(bboxes_num, 1);
        float *data = bboxes_res + resCount * 8;
        data[0] = bboxes[i * 7];
        data[1] = bboxes[i * 7 + 1];
        data[2] = bboxes[i * 7 + 2];
        data[3] = bboxes[i * 7 + 3];
        data[4] = bboxes[i * 7 + 4];
        data[5] = bboxes[i * 7 + 5];
        data[6] = bboxes[i * 7 + 6];
        bboxes_labels[resCount] = labels[i];
        data[7] = scores[i];
    }
}

__global__ void softmax_kernel(const float* semantic,
                                  int grid_x_size, int grid_y_size, 
                                  int num_class,
                                  int* seg_res, float* seg_scores) {
    int x = blockIdx.x;
    int y = threadIdx.x;
    int index = y * grid_x_size + x;

    float score = 0.;
    int label = 0;
    int dev_out_index;
    for (int i = 0; i < num_class; ++i) {
        dev_out_index = i * grid_x_size * grid_y_size + index;
        if (semantic[dev_out_index + i] > score) {
            score = semantic[dev_out_index];
            label = i;
        }
    }

    seg_scores[index] = score;
    seg_res[index] = label;
}

cudaError_t postprocess_launch(const float *bboxes, 
                      const float *scores, 
                      const int *labels, 
                      const float score_thresh,
                      float *bboxes_res, int* bboxes_labels,
                      int *bboxes_num,
                      cudaStream_t stream)
{
  dim3 blocks(1000 / THREADS, 1, 1);
  if (1000 % THREADS != 0) {
    blocks.x += 1;
  }
  dim3 threads(THREADS, 1, 1);

  postprocess_kernal<<<blocks, threads, 0, stream>>>
                (bboxes, scores, labels, score_thresh, bboxes_res, bboxes_labels, bboxes_num);
  return cudaGetLastError();
}

cudaError_t seg_postprocess_launch(const float *semantic, 
                                   int grid_x_size, int grid_y_size,
                                   int num_class, int *seg_res, float *seg_scores,
                                   cudaStream_t stream) {

    dim3 blocks(grid_x_size, 1, 1);
    dim3 threads(grid_y_size, 1, 1);
    softmax_kernel<<<blocks, threads, 0, stream>>>(semantic, grid_x_size, grid_y_size, num_class, seg_res, seg_scores);    
    return cudaGetLastError();
}