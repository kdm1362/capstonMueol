cmake_minimum_required(VERSION 3.5)

project(image_tools)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(darknet_ros_msgs REQUIRED)

include_directories(include)

add_library(imagetools SHARED
  src/showimage.cpp)

target_compile_definitions(imagetools
  PRIVATE "IMAGE_TOOLS_BUILDING_DLL")
ament_target_dependencies(imagetools
  "rclcpp"
  "sensor_msgs"
  "std_msgs"
  "darknet_ros_msgs"
  "rclcpp_components"
  "OpenCV")
  
rclcpp_components_register_node(imagetools PLUGIN "image_tools::ShowImage" EXECUTABLE showimage)

install(TARGETS
  imagetools
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  find_package(rmw_implementation_cmake REQUIRED)

endif()

ament_package()
