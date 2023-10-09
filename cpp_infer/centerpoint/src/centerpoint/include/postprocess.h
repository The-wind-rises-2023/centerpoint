
#ifndef POSTPROCESS_H_
#define POSTPROCESS_H_

#include <vector>
#include "kernel.h"
/*
box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
*/
struct Bndbox {
    int id;
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float rt;
    int label;
    float score;
    Bndbox(){};
    Bndbox(int id_, float x_, float y_, float z_, float w_, float l_, float h_, float rt_, int label_, float score_)
        : id(id_), x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), rt(rt_), label(label_), score(score_) {}
};

const std::vector<std::vector<int> > label_color {
    std::vector<int>{0, 0, 0}, // kong
    std::vector<int>{220, 220, 220}, // unlabel
    std::vector<int>{128, 64, 128}, // road
    std::vector<int>{244, 35, 232}, // station
    std::vector<int>{70, 70, 70}, // fence
};

struct SemanticBev {
    int grid_x_size;
    int grid_y_size;
    std::vector<int> labels;
    std::vector<float> scores;

    SemanticBev() {};
    SemanticBev(const int in_grid_x_size, const int in_grid_y_size, const int *in_label, const float *in_score) 
       : grid_x_size(in_grid_x_size), grid_y_size(in_grid_y_size) {
      size_t sz = in_grid_x_size * in_grid_y_size;
      labels.resize(sz);
      scores.resize(sz);
      labels = std::vector<int>(in_label, in_label + sz);
      scores = std::vector<float>(in_score, in_score + sz);
    }
    // void show() {
    //   cv::Mat src = cv::Mat(grid_x_size, grid_y_size, CV_8UC3);
    //   for (size_t i = 0; i < labels.size(); ++i) {
    //     cv::Vec3b &color = src.at<cv::Vec3b>(i / grid_y_size, i % grid_y_size);
    //     color[0] = label_color[labels[i]][0];
    //     color[1] = label_color[labels[i]][1];
    //     color[2] = label_color[labels[i]][2];
    //   }
    //   cv::imshow("SemanticBev", src);
    //   cv::waitKey(1);
    // }
};

class PostProcessCuda {
  private:
    Params params_;
    cudaStream_t stream_ = 0;

    float *h_bboxes_res_ = nullptr;
    int *h_bboxes_labels_ = nullptr;
    int *h_bboxes_num_ = nullptr;

    int *h_seg_res_ = nullptr;
    float *h_seg_scores_ = nullptr;

  public:
    PostProcessCuda(cudaStream_t stream = 0);
    ~PostProcessCuda();

    int doPostprocessCuda(const float *bboxes, const float *scores, const int *labels, const float *semantic, 
                          std::vector<Bndbox> &res, SemanticBev &seg_bev);
    int detPostprocess(const float *bboxes, const float *scores, const int *labels, float *bboxes_res, int *bboxes_label, int *bboxes_num);
    int segPostprocess(const float *semantic, int *seg_res, float *seg_scores);

};

int nms_cpu(std::vector<Bndbox> bndboxes, const float nms_thresh, std::vector<Bndbox> &nms_pred);

#endif
