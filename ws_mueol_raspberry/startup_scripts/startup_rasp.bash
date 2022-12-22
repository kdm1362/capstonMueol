#!/bin/bash

source /opt/ros/foxy/setup.bash
source /home/ubuntu/ws_mueol/install/setup.bash

export ROS_DOMAIN_ID=7
ros2 launch darknet_ros yolov4-tiny.launch.py&

