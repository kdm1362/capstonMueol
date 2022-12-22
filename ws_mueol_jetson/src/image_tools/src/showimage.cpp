// Copyright 2015 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "darknet_ros_msgs/msg/bounding_boxes.hpp"

#include "image_tools/visibility_control.h"

#include "./policy_maps.hpp"


using namespace cv;

// cuda sgm
class CudaSGM
{
public:
  Mat left, right;
  Mat disparity;
  Mat disp_8u, colored;
  rclcpp::Time time_last_process;
  rclcpp::Clock* clock_ros;
  //clibration
  Mat frame2;
  Mat map_l1, map_l2, map_r1, map_r2;
  
  cuda::GpuMat d_left, d_right, d_disp;
  Ptr<cuda::StereoSGM> sgm;
  Ptr<cuda::StereoBeliefPropagation> bp;
  Ptr<cuda::StereoConstantSpaceBP> csbp;
  
  int64 work_begin;
  double work_fps;
  
public:
  CudaSGM()
  {
    clock_ros = new rclcpp::Clock();
    time_last_process = clock_ros->now();
    sgm = cuda::createStereoSGM(0, 256);
    disparity.create(Size(640,480), CV_8U);
    cuda::GpuMat d_disp(Size(640,480), CV_8U);
    
    //calib
    Size imageSize = Size(640, 480);
    
    // left calibration info
    Mat cameraMatrix_l = (Mat1d(3,3) <<
    576.995640, 0.000000, 299.722497, 
    0.000000, 578.712143, 242.219288, 
    0.000000, 0.000000, 1.000000
    );

    Mat distCoeffs_l = (Mat1d(1, 5) << -0.430588, 0.215187, 0.005551, -0.000214, 0.000000);
    initUndistortRectifyMap(cameraMatrix_l, distCoeffs_l, Mat(), cameraMatrix_l, imageSize, CV_32FC1, map_l1, map_l2);
    
    // right calibration info
    Mat cameraMatrix_r = (Mat1d(3,3) <<
    576.995640, 0.000000, 299.722497, 
    0.000000, 578.712143, 242.219288, 
    0.000000, 0.000000, 1.000000
    );

    Mat distCoeffs_r = (Mat1d(1,5) << -0.430588, 0.215187, 0.005551, -0.000214, 0.000000);
    initUndistortRectifyMap(cameraMatrix_r, distCoeffs_r, Mat(), cameraMatrix_r, imageSize, CV_32FC1, map_r1, map_r2);
  }
  
  Mat get_disparity(){
    if((clock_ros->now() - time_last_process).seconds()*1000 < 60)
      return disparity;
    if(left.empty() || right.empty())
      return Mat(640, 480, CV_8UC1);
      
    cameraCalib(left, true);
    cameraCalib(right, false);

    d_left.upload(left);
    d_right.upload(right);
    
    sgm->compute(d_left, d_right, d_disp);
    d_disp.download(disparity);
    
    disparity.convertTo(disparity, CV_32F);
    disparity = disparity * 256. / (256 * 16);
    disparity.convertTo(disp_8u, CV_8UC1);
    
    imshow("disp", disp_8u);
    
    time_last_process = clock_ros->now();
    return disp_8u;
  }
  
  void update_image_left(Mat cvframe){
  //test swap left, right
    cvtColor(cvframe, left, COLOR_BGR2GRAY);
  }
  
  void update_image_right(Mat cvframe){
    cvtColor(cvframe, right, COLOR_BGR2GRAY);
  }
  void cameraCalib(Mat& frame, bool isLeft){
    if(isLeft)
      remap(frame, frame2, map_l1, map_l2, INTER_LINEAR);
    else
      remap(frame, frame2, map_r1, map_r2, INTER_LINEAR);
    resize(frame2, frame2, Size(640, 480));
    frame2.convertTo(frame, CV_8UC1);
  }
};
// cuda sgm


namespace image_tools
{
class ShowImage : public rclcpp::Node
{
  darknet_ros_msgs::msg::BoundingBoxes boxes;
public:
  IMAGE_TOOLS_PUBLIC
  explicit ShowImage(const rclcpp::NodeOptions & options)
  : Node("showimage", options)
  {
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    parse_parameters();
    initialize();
  }

private:
  CudaSGM sgbm;
  cv::Mat disp8;
  
  IMAGE_TOOLS_LOCAL
  void initialize()
  {
    if (show_image_) {
      // Initialize an OpenCV named window called "showimage".
      //cv::namedWindow("left", cv::WINDOW_AUTOSIZE);
      //cv::namedWindow("right", cv::WINDOW_AUTOSIZE);
    }
    // Set quality of service profile based on command line options.
    auto qos = rclcpp::QoS(
      rclcpp::QoSInitialization(
        // The history policy determines how messages are saved until taken by
        // the reader.
        // KEEP_ALL saves all messages until they are taken.
        // KEEP_LAST enforces a limit on the number of messages that are saved,
        // specified by the "depth" parameter.
        history_policy_,
        // Depth represents how many messages to store in history when the
        // history policy is KEEP_LAST.
        depth_
    ));
    // The reliability policy can be reliable, meaning that the underlying transport layer will try
    // ensure that every message gets received in order, or best effort, meaning that the transport
    // makes no guarantees about the order or reliability of delivery.
    qos.reliability(reliability_policy_);
    auto callback_l = [this](const sensor_msgs::msg::Image::SharedPtr msg)
      {
        process_image(msg, show_image_, true);
      };
    auto callback_r = [this](const sensor_msgs::msg::Image::SharedPtr msg)
      {
        process_image(msg, show_image_, false);
      };
    auto callback_Boxes = [this](const darknet_ros_msgs::msg::BoundingBoxes::SharedPtr msg)
      {
        process_Boxes(msg);
      };

    RCLCPP_INFO(this->get_logger(), "Subscribing to topic '%s'", topic_left_.c_str());
    RCLCPP_INFO(this->get_logger(), "Subscribing to topic '%s'", topic_right_.c_str());    
    sub_l = create_subscription<sensor_msgs::msg::Image>(topic_left_, qos, callback_l);
    sub_r = create_subscription<sensor_msgs::msg::Image>(topic_right_, qos, callback_r);
    sub_Boxes = create_subscription<darknet_ros_msgs::msg::BoundingBoxes>("darknet_ros/bounding_boxes", qos, callback_Boxes);
  }

  IMAGE_TOOLS_LOCAL
  void parse_parameters()
  {
    // Parse 'reliability' parameter
    rcl_interfaces::msg::ParameterDescriptor reliability_desc;
    reliability_desc.description = "Reliability QoS setting for the image subscription";
    reliability_desc.additional_constraints = "Must be one of: ";
    for (auto entry : name_to_reliability_policy_map) {
      reliability_desc.additional_constraints += entry.first + " ";
    }
    const std::string reliability_param = this->declare_parameter(
      "reliability", "reliable", reliability_desc);
    auto reliability = name_to_reliability_policy_map.find(reliability_param);
    if (reliability == name_to_reliability_policy_map.end()) {
      std::ostringstream oss;
      oss << "Invalid QoS reliability setting '" << reliability_param << "'";
      throw std::runtime_error(oss.str());
    }
    reliability_policy_ = reliability->second;

    // Parse 'history' parameter
    rcl_interfaces::msg::ParameterDescriptor history_desc;
    history_desc.description = "History QoS setting for the image subscription";
    history_desc.additional_constraints = "Must be one of: ";
    for (auto entry : name_to_history_policy_map) {
      history_desc.additional_constraints += entry.first + " ";
    }
    const std::string history_param = this->declare_parameter(
      "history", name_to_history_policy_map.begin()->first, history_desc);
    auto history = name_to_history_policy_map.find(history_param);
    if (history == name_to_history_policy_map.end()) {
      std::ostringstream oss;
      oss << "Invalid QoS history setting '" << history_param << "'";
      throw std::runtime_error(oss.str());
    }
    history_policy_ = history->second;

    // Declare and get remaining parameters
    depth_ = this->declare_parameter("depth", 10);
    show_image_ = this->declare_parameter("show_image", true);
  }

  /// Convert a sensor_msgs::Image encoding type (stored as a string) to an OpenCV encoding type.
  /**
   * \param[in] encoding A string representing the encoding type.
   * \return The OpenCV encoding type.
   */
  IMAGE_TOOLS_LOCAL
  int encoding2mat_type(const std::string & encoding)
  {
    if (encoding == "mono8") {
      return CV_8UC1;
    } else if (encoding == "bgr8") {
      return CV_8UC3;
    } else if (encoding == "mono16") {
      return CV_16SC1;
    } else if (encoding == "rgba8") {
      return CV_8UC4;
    } else if (encoding == "bgra8") {
      return CV_8UC4;
    } else if (encoding == "32FC1") {
      return CV_32FC1;
    } else if (encoding == "rgb8") {
      return CV_8UC3;
    } else {
      throw std::runtime_error("Unsupported encoding type");
    }
  }

  /// Convert the ROS Image message to an OpenCV matrix and display it to the user.
  // \param[in] msg The image message to show.
  IMAGE_TOOLS_LOCAL
  void process_image(
    const sensor_msgs::msg::Image::SharedPtr msg, bool show_image, bool left)
  {
    if (show_image) {
      // Convert to an OpenCV matrix by assigning the data.
      cv::Mat frame(
        msg->height, msg->width, encoding2mat_type(msg->encoding),
        const_cast<unsigned char *>(msg->data.data()), msg->step);

      if (msg->encoding == "rgb8") {
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
      }
      
      cv::Mat cvframe = frame;

      // Show the image in a window called "showimage".
      std::string whichimage;
      if(left){
        whichimage="left";
        sgbm.update_image_left(cvframe);
      } else{
        whichimage="right";
        sgbm.update_image_right(cvframe);
        // right image is remote so if right image arrive then get disparity
        //sgbm.get_disparity().copyTo(disp8);
        sgbm.get_disparity();
      }
      
      // cv::imshow(whichimage, cvframe);
      // Draw the screen and wait for 1 millisecond.
      cv::waitKey(1);
    }
  }
  
  IMAGE_TOOLS_LOCAL
  void process_Boxes(const darknet_ros_msgs::msg::BoundingBoxes::SharedPtr msg){
    boxes = *msg;
    for(auto box: boxes.bounding_boxes){
      if (box.id == 0){
        //target!!
        // double dist = get_distance(i.xmax-i.xmin, i.ymax-i.ymin);
        int width = box.xmax-box.xmin;
        int height = box.ymax-box.ymin;
        int cx = box.xmin + int(width/2);
        int cy = box.ymin + int(height/2);
        
        // most frequncy data in box
        int count = 0;
        double dist = 0.f;
        for (int i=box.ymin; i<box.ymax; i++){
          for(int j=box.xmin; j<box.xmax; j++){
            int val = sgbm.disp_8u.data[640*i + j];
            if (val==0)
              continue;
            dist += val;
            count++;
          }
        }
        
        dist /= count;
        
        if(dist>218)
          dist = 30;
        else if(dist>160)
          dist = 40;
        else if(dist>120)
          dist = 50;
        else if(dist>100)
          dist = 60;
        else if(dist>85)
          dist = 70;
        else if(dist>80)
          dist = 80;
        else if(dist>75)
          dist = 90;
        else if(dist>70)
          dist = 100;
        else
          dist = 300;
        
        //cv::imshow("debug", sgbm.disp_8u);
        //int dist = int(disp8.data[640*cy + cx]);
        std::cout << dist << std::endl;
        // pick_trash(
        std::cout << box.xmin << ", " << box.xmax << std::endl;
      }
    }
  }
  

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_l;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_r;
  rclcpp::Subscription<darknet_ros_msgs::msg::BoundingBoxes>::SharedPtr sub_Boxes;
  size_t depth_ = rmw_qos_profile_default.depth;
  rmw_qos_reliability_policy_t reliability_policy_ = rmw_qos_profile_default.reliability;
  rmw_qos_history_policy_t history_policy_ = rmw_qos_profile_default.history;
  bool show_image_ = true;
  std::string topic_left_ = "camera_left";
  std::string topic_right_ = "camera_right";
};

}  // namespace image_tools

RCLCPP_COMPONENTS_REGISTER_NODE(image_tools::ShowImage)
