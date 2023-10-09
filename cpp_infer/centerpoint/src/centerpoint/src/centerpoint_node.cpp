#include <ros/ros.h>
#include <ros/publisher.h>
#include <string>

// ros messages
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <cuda_runtime.h>

#include "params.h"
#include "centerpoint.h"
#include <iostream>
#include <sstream>
#include <fstream>

#include <tf/tf.h>

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

std::string Data_File = "src/centerpoint/data/";
std::string Save_Dir = "src/centerpoint/eval/";
std::string Model_File = "src/centerpoint/model/end2end";

class CloudSubscriber
{
protected:
  ros::NodeHandle nh;
  ros::Subscriber sub;
  ros::Publisher pub;
  cudaEvent_t start;
  cudaEvent_t stop;
  float elapsedTime{0.0f};
  cudaStream_t stream{NULL};
  std::shared_ptr<PointPillar> _pointpillar;
  float *_points_data = nullptr;
  std::vector<Bndbox> _nms_pred;

private:
  sensor_msgs::PointCloud2 _cloud;
  bool _is_start{false};

public:
  CloudSubscriber()
  {
    sub = nh.subscribe<sensor_msgs::PointCloud2>("/iv_points_arm", 5, &CloudSubscriber::subCallback, this);
    pub = nh.advertise<visualization_msgs::MarkerArray>("/centerpoint_obs", 10);

    _pointpillar.reset(new PointPillar(Model_File, stream));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaStreamCreate(&stream));

  }

  ~CloudSubscriber()
  {
    sub.shutdown();
    checkCudaErrors(cudaFree(_points_data));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));
  }

  void subCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
  {

    if (!_is_start) 
    {  
      ROS_INFO("Begin subscribe data!");
      _is_start = true;
    }

    if (msg->data.empty())
    {
      ROS_WARN("Received an empty cloud message. Skipping further processing");
      return;
    }
    _cloud = *msg;
    publish();
  }
  
  void publish()
  {
    if (!_is_start) 
    {
      ROS_WARN("Not start. please check input data");
      return;
    }
    ROS_INFO_STREAM("Received a cloud message with " << _cloud.height * _cloud.width << " points");

    size_t point_num = _cloud.height * _cloud.width;

    // ROS_INFO_STREAM("width:\t"<<(int)_cloud.width);
    // ROS_INFO_STREAM("height:\t"<<(int)_cloud.height);
    // ROS_INFO_STREAM("point_step:\t"<<(int)_cloud.point_step);
    // ROS_INFO_STREAM("row_step:\t"<<(int)_cloud.row_step);
    // ROS_INFO_STREAM("fields_size:\t"<<(int)_cloud.fields.size());
    // for (size_t i = 0; i < _cloud.fields.size(); ++i) 
    // {
    //   ROS_INFO_STREAM("fields\t" << _cloud.fields[i].offset << "\t" << _cloud.fields[i].datatype << "\t" << _cloud.fields[i].name);
    // }

    float *data = (float*)malloc(point_num * 4 * sizeof(float));

    float *px = data + 0;
    float *py = data + 1;
    float *pz = data + 2;
    float *pi = data + 3;    

    for(size_t i = 0; i < point_num; ++i)
    {
      std::memcpy(pz, &_cloud.data[32 * i], 4);
      std::memcpy(py, &_cloud.data[32 * i + 4], 4);
      *py = -*py;
      std::memcpy(px, &_cloud.data[32 * i + 8], 4);
      std::memcpy((int*)pi, &_cloud.data[32 * i + 24], 2);
      *pi = _cloud.data[32 * i + 24];

      px += 4; py += 4; pz += 4; pi += 4;

    }

    //test data extract
    // px = data + 0; py = data + 1; pz = data + 2; pi = data + 3;
    
    // for (size_t i = 0; i < point_num; ++i)
    // {
    //   ROS_INFO_STREAM("Point " << i << "\tinfo is\t" << 
    //                   *px << "\t" << 
    //                   *py << "\t" << 
    //                   *pz << "\t" << 
    //                   *pi );
    //   px += 4; py += 4; pz += 4; pi += 4;
    // }

    visualization_msgs::MarkerArray output_msg;

    // model process
    _nms_pred.clear();
    process(data, point_num, _nms_pred);

    ROS_INFO_STREAM("Get obstacles succeed! num is : " << (int)_nms_pred.size());
    for (const auto &bb_obs : _nms_pred)
    {
      // add obs to msgs
      visualization_msgs::Marker bb_msg;

      bb_msg.header.frame_id = "innovusion";
      bb_msg.header.stamp = ros::Time::now();
      bb_msg.type = visualization_msgs::Marker::CUBE;
      bb_msg.ns = "obstacles";
      bb_msg.pose.position.x = bb_obs.z;
      bb_msg.pose.position.y = -bb_obs.y;
      bb_msg.pose.position.z = bb_obs.x;

      bb_msg.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(bb_obs.rt, 0, 0);

      bb_msg.id = bb_obs.id;
      bb_msg.color.a = 0.5;
      bb_msg.color.g = 1.0;

      bb_msg.scale.x = bb_obs.h;
      bb_msg.scale.y = bb_obs.l;
      bb_msg.scale.z = bb_obs.w;

      output_msg.markers.push_back(bb_msg);
    }
    pub.publish(output_msg);
    
  }

  void process(float* points_data, size_t points_num, std::vector<Bndbox>& nms_pred) {

    nms_pred.reserve(100);

    unsigned int points_data_size = points_num * 4 * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **)&_points_data, points_data_size));
    checkCudaErrors(cudaMemcpy(_points_data, points_data, points_data_size, cudaMemcpyDefault));
    checkCudaErrors(cudaDeviceSynchronize());
#if PERFORMANCE_LOG
    cudaEventRecord(start, stream);
    std::cout << "point data: " << points_num << std::endl;
#endif
    // PointPillar pointpillar(Model_File, stream);
    _pointpillar->doinfer(_points_data, points_num, nms_pred);
#if PERFORMANCE_LOG
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout<<"TIME: pointpillar: "<< elapsedTime <<" ms." <<std::endl;
#endif
    
  }

};

int main(int argc, char* argv[])
{
  ros::init (argc, argv, "centerpoint");
  ros::NodeHandle nh;

  CloudSubscriber c;

  while(nh.ok()) 
  {
    ros::spin();
  }
  return 0;
}
