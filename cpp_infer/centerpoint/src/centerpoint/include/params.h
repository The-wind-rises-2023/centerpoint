
#ifndef PARAMS_H_
#define PARAMS_H_
const int MAX_VOXELS = 40000;
class Params
{
  public:
    static const int num_classes = 5;
    static const int seg_num_classes = 5;
    const char *class_name [num_classes] = { "Car", "Pedestrian", "Cyclist", "Truck", "Train",};
    const char *seg_class_name [seg_num_classes] = { "Kong", "Unlabel", "Road", "Station", "Fence",};
    const float min_x_range = 0.0;
    const float max_x_range = 120.;
    const float min_y_range = -20.;
    const float max_y_range = 20.;
    const float min_z_range = -4.0;
    const float max_z_range = 4.0;
    // the size of a pillar
    const float pillar_x_size = 0.1;
    const float pillar_y_size = 0.1;
    const float pillar_z_size = 8;
    const int max_num_points_per_pillar = 32;
    const int num_point_values = 4;
    // the number of feature maps for pillar scatter
    const int num_feature_scatter = 64;

    // the score threshold for classification
    const float score_thresh = 0.3;
    const float nms_thresh = 0.2;
    const int max_num_pillars = MAX_VOXELS;
    const int pillarPoints_bev = max_num_points_per_pillar * max_num_pillars;
    // the detected boxes result decode by (x, y, z, w, l, h, yaw)
    const int num_box_values = 7;
    // the input size of the 2D backbone network
    const int grid_x_size = (max_x_range - min_x_range) / pillar_x_size;
    const int grid_y_size = (max_y_range - min_y_range) / pillar_y_size;
    const int grid_z_size = (max_z_range - min_z_range) / pillar_z_size;

    // todo: fix bboxes dimension
    const int bboxes_dim1 = 1; 
    const int booxes_dim2 = 1000;
    const int booxes_dim3 = 7;
    const int scores_dim1 = 1;
    const int scores_dim2 = 1000;
    const int labels_dim1 = 1;
    const int labels_dim2 = 1000;

    // the output size of the 2D backbone network
    // const int feature_x_size = grid_x_size / 2;
    // const int feature_y_size = grid_y_size / 2;
    Params() {};
};
#endif
