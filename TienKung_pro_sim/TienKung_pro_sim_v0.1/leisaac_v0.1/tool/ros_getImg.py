#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 node that subscribes to an RGB camera topic and saves the latest
image to a single JPG file at 10Hz (overwriting the same file).

Usage:
  - Source your ROS 2 environment, then run:
  ROS_DOMAIN_ID=146 python3 deployment/rgb_act/ros_getImg.py

Parameters (can be changed in code or via ROS param tools):
  - camera_topic: topic to subscribe to (default: '/camera/color/image_raw')
  - save_path: path to the output JPG file (default: './camera.jpg')

Only RGB images are saved; depth images are ignored.
"""

import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Optional


class RosGetImg(Node):
    def __init__(self):
        super().__init__('ros_get_img_node')

        # Parameters (can be overridden later via ROS param or code)
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('save_path', './camera.jpg')
        self.declare_parameter('save_hz', 10.0)

        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.save_path = self.get_parameter('save_path').get_parameter_value().string_value
        self.save_hz = float(self.get_parameter('save_hz').get_parameter_value().double_value)

        # Ensure output directory exists
        out_dir = os.path.dirname(os.path.abspath(self.save_path))
        if out_dir and not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                self.get_logger().error(f'Unable to create output directory {out_dir}: {e}')

        # CV bridge
        self.bridge = CvBridge()

        # Latest RGB image (numpy array in RGB order)
        self.latest_rgb: Optional[np.ndarray] = None

        # Subscribe to camera topic
        self.get_logger().info(f'Subscribing to camera topic: {self.camera_topic}')
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10)

        # Timer to save image at desired frequency
        period = 1.0 / max(0.0001, float(self.save_hz))
        self.get_logger().info(f'Saving to {self.save_path} at {self.save_hz} Hz (period {period:.3f}s)')
        self.timer = self.create_timer(period, self.timer_callback)

    def image_callback(self, msg: Image):
        try:
            # Convert to RGB8
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            # Store as numpy array (RGB order)
            self.latest_rgb = cv_image
        except Exception as e:
            self.get_logger().error(f'Failed to convert incoming image: {e}')

    def timer_callback(self):
        if self.latest_rgb is None:
            self.get_logger().debug('No image received yet; skipping save')
            return

        try:
            bgr = cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.save_path, bgr) 
        except Exception as e:
            self.get_logger().error(f'Unexpected error saving image to {self.save_path}: {e}')


def main(argv=None):
    rclpy.init(args=argv)
    node = RosGetImg()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down ros_get_img node')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)
