#!/bin/bash
# Check dependencies
echo "Checking dependencies..."

if ! dpkg -s libzmq3-dev &> /dev/null; then
    echo "Installing libzmq3-dev..."
    sudo apt-get update && sudo apt-get install -y libzmq3-dev
fi

if ! dpkg -s nlohmann-json3-dev &> /dev/null; then
    echo "Installing nlohmann-json3-dev..."
    # Try generic package name or download header if failing
    if ! sudo apt-get install -y nlohmann-json3-dev; then
         echo "Attempting to install 'nlohmann-json-dev' instead..."
         sudo apt-get install -y nlohmann-json-dev
    fi
fi

# Source ROS 2 (Assume Humble or generic environment)
if [ -z "$ROS_DISTRO" ]; then
    echo "Sourcing ROS 2 Humble..."
    source /opt/ros/humble/setup.bash
fi

echo "Compiling zmq_image_bridge..."

# Compile command
# Uses -I for Includes and -L -l for linking ROS2 libraries and ZMQ
g++ zmq_image_bridge.cpp -o zmq_image_bridge \
    -std=c++17 \
    -O3 \
    -I/opt/ros/$ROS_DISTRO/include/rclcpp \
    -I/opt/ros/$ROS_DISTRO/include/sensor_msgs \
    -I/opt/ros/$ROS_DISTRO/include/std_msgs \
    -I/opt/ros/$ROS_DISTRO/include/builtin_interfaces \
    -I/opt/ros/$ROS_DISTRO/include/rcl \
    -I/opt/ros/$ROS_DISTRO/include/rcutils \
    -I/opt/ros/$ROS_DISTRO/include/rcl_yaml_param_parser \
    -I/opt/ros/$ROS_DISTRO/include/libyaml_vendor \
    -I/opt/ros/$ROS_DISTRO/include/rmw \
    -I/opt/ros/$ROS_DISTRO/include/rosidl_runtime_c \
    -I/opt/ros/$ROS_DISTRO/include/rosidl_runtime_cpp \
    -I/opt/ros/$ROS_DISTRO/include/rosidl_typesupport_interface \
    -I/opt/ros/$ROS_DISTRO/include/tracetools \
    -I/opt/ros/$ROS_DISTRO/include/rcl_interfaces \
    -I/opt/ros/$ROS_DISTRO/include/rcpputils \
    -I/opt/ros/$ROS_DISTRO/include/libstatistics_collector \
    -I/opt/ros/$ROS_DISTRO/include/statistics_msgs \
    -I/opt/ros/$ROS_DISTRO/include/rosgraph_msgs \
    -L/opt/ros/$ROS_DISTRO/lib \
    -lrclcpp -lrcutils -lrcl -lrcpputils -lrmw -lrmw_implementation \
    -llibstatistics_collector -lrcl_yaml_param_parser -lament_index_cpp \
    -lrcl_logging_interface -lrcl_logging_spdlog \
    -lrosidl_runtime_c \
    -lrosidl_typesupport_cpp -lrosidl_typesupport_c \
    -lrosidl_typesupport_introspection_cpp -lrosidl_typesupport_introspection_c \
    -lrcl_interfaces__rosidl_typesupport_cpp -lrcl_interfaces__rosidl_typesupport_c -lrcl_interfaces__rosidl_generator_c \
    -lrosgraph_msgs__rosidl_typesupport_cpp -lstatistics_msgs__rosidl_typesupport_cpp \
    -lsensor_msgs__rosidl_typesupport_cpp -ltracetools \
    -lzmq -lpthread -latomic

if [ $? -eq 0 ]; then
    echo "Build successful! Run with: ./zmq_image_bridge"
else
    echo "Build failed. Please check errors."
    exit 1
fi
