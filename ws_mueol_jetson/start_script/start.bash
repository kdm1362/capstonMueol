#!/bin/bash

source /opt/ros/foxy/setup.bash
source ~/ws_mueol_jetson/install/setup.bash

ros2 run v4l2_camera v4l2_camera_node&
ros2 run image_tools showimage&

