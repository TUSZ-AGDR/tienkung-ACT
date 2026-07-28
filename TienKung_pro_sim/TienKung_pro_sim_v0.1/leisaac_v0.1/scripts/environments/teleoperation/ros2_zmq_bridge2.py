#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS 2 ZMQ Bridge for Tiangong Pro Simulation Closed-Loop Control.

Architecture:
  - BIND tcp://*:5555 (PUB)  -> receives ROS2 arm/hand cmds, forwards to Isaac Lab
  - CONNECT to {sim_host}:5556 (SUB) -> receives robot status from Isaac Lab,
    publishes to ROS2
  - CONNECT to {sim_host}:5557 (SUB) -> receives camera images from Isaac Lab,
    publishes to ROS2

Image Protocol (multi-part ZMQ from Isaac Lab):
  Part 0: JSON metadata  {"width": 640, "height": 480, "format": "raw"}
  Part 1: Raw RGB bytes   (W*H*3, rgb8)
  Part 2: Raw Depth bytes  (W*H*2, 16UC1 mm) [optional]

Usage:
  python ros2_zmq_bridge.py --sim-host host.docker.internal
  python ros2_zmq_bridge.py --sim-host 192.168.1.100
"""

import argparse
import json
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image

try:
    from bodyctrl_msgs.msg import MotorStatusMsg, MotorStatus, CmdSetMotorPosition
except ImportError:
    print("[ERROR] bodyctrl_msgs not found. Please source the correct workspace.")
    sys.exit(1)

import zmq

try:
    import cv2
    from cv_bridge import CvBridge
except ImportError:
    print("[ERROR] opencv-python or cv_bridge not found.")
    sys.exit(1)


class ZmqImageReceiver:
    """Receives multi-part image messages from Isaac Lab via ZMQ SUB,
    converts to ROS sensor_msgs/Image, and publishes."""

    def __init__(self, sim_host: str, img_port: int, node: Node):
        self._node = node
        self._logger = node.get_logger()
        self._bridge = CvBridge()

        self._zmq_context = zmq.Context()
        self._socket = self._zmq_context.socket(zmq.SUB)
        addr = f"tcp://{sim_host}:{img_port}"
        self._socket.connect(addr)
        self._socket.setsockopt(zmq.RCVHWM, 2)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._logger.info(f"[IMG] ZMQ SUB connected to {addr}")

        self._pub_rgb = node.create_publisher(Image, "/sim/camera_head/color/image_raw", 10)
        self._pub_depth = node.create_publisher(Image, "/sim/camera_head/depth/image_raw", 10)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """Continuously receive and publish images."""
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)

        while self._running and rclpy.ok():
            socks = dict(poller.poll(timeout=100))
            if self._socket not in socks:
                continue

            try:
                meta_raw = self._socket.recv(flags=zmq.NOBLOCK)
                meta = json.loads(meta_raw)
                w, h = meta.get("width", 640), meta.get("height", 480)

                if not self._socket.getsockopt(zmq.RCVMORE):
                    continue
                rgb_raw = self._socket.recv(flags=zmq.NOBLOCK)

                depth_raw = None
                if self._socket.getsockopt(zmq.RCVMORE):
                    depth_raw = self._socket.recv(flags=zmq.NOBLOCK)

                self._publish_rgb(rgb_raw, w, h)
                if depth_raw:
                    self._publish_depth(depth_raw, w, h)

            except zmq.Again:
                pass
            except Exception as e:
                self._logger.error(f"[IMG] Error: {e}", throttle_duration_sec=5.0)

    def _publish_rgb(self, raw_bytes: bytes, w: int, h: int):
        """Convert raw RGB bytes to ROS Image and publish."""
        expected = w * h * 3
        if len(raw_bytes) != expected:
            self._logger.warning(
                f"[IMG] RGB size mismatch: got {len(raw_bytes)}, expected {expected}",
                throttle_duration_sec=10.0,
            )
            return

        np_img = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3))
        ros_img = self._bridge.cv2_to_imgmsg(np_img, encoding="rgb8")
        ros_img.header.stamp = self._node.get_clock().now().to_msg()
        ros_img.header.frame_id = "sim_camera_head"
        self._pub_rgb.publish(ros_img)

    def _publish_depth(self, raw_bytes: bytes, w: int, h: int):
        """Convert raw depth bytes to ROS Image and publish."""
        expected = w * h * 2
        if len(raw_bytes) != expected:
            self._logger.warning(
                f"[IMG] Depth size mismatch: got {len(raw_bytes)}, expected {expected}",
                throttle_duration_sec=10.0,
            )
            return

        np_depth = np.frombuffer(raw_bytes, dtype=np.uint16).reshape((h, w))
        ros_depth = self._bridge.cv2_to_imgmsg(np_depth, encoding="16UC1")
        ros_depth.header.stamp = self._node.get_clock().now().to_msg()
        ros_depth.header.frame_id = "sim_camera_head"
        self._pub_depth.publish(ros_depth)

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._socket.close()
        self._zmq_context.term()


class TiangongRosBridge(Node):
    """ROS 2 node that bridges sensor streams between Isaac Lab and the ACT
    policy deployment stack via ZMQ."""

    def __init__(
        self,
        sim_host: str = "host.docker.internal",
        cmd_port: int = 6555,
        status_port: int = 6556,
        image_port: int = 6557,
        debug_cmd: bool = False,
    ):
        self._debug_cmd = debug_cmd
        super().__init__("tiangong_ros_bridge")

        self._zmq_context = zmq.Context()

        self.cmd_socket = self._zmq_context.socket(zmq.PUB)
        self.cmd_socket.connect(f"tcp://{sim_host}:{cmd_port}")
        self.get_logger().info(f"[CMD] PUB connected to tcp://{sim_host}:{cmd_port}")

        self.status_socket = self._zmq_context.socket(zmq.SUB)
        self.status_socket.connect(f"tcp://{sim_host}:{status_port}")
        self.status_socket.setsockopt(zmq.RCVHWM, 1)
        self.status_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.get_logger().info(f"[STATUS] SUB connected to tcp://{sim_host}:{status_port}")

        self._img_receiver = ZmqImageReceiver(sim_host, image_port, self)

        self.ID_HEAD = [1, 2, 3]
        self.ID_ARM_L = [11, 12, 13, 14, 15, 16, 17]
        self.ID_ARM_R = [21, 22, 23, 24, 25, 26, 27]
        self.ID_WAIST = [31]
        self.ID_LEG_L = [51, 52, 53, 54, 55, 56]
        self.ID_LEG_R = [61, 62, 63, 64, 65, 66]

        self.ID_TO_NAME = {
            1: "head_roll_joint",  2: "head_pitch_joint",  3: "head_yaw_joint",
            11: "shoulder_pitch_l_joint", 12: "shoulder_roll_l_joint",
            13: "shoulder_yaw_l_joint",
            14: "elbow_pitch_l_joint",    15: "elbow_yaw_l_joint",
            16: "wrist_pitch_l_joint",    17: "wrist_roll_l_joint",
            21: "shoulder_pitch_r_joint", 22: "shoulder_roll_r_joint",
            23: "shoulder_yaw_r_joint",
            24: "elbow_pitch_r_joint",    25: "elbow_yaw_r_joint",
            26: "wrist_pitch_r_joint",    27: "wrist_roll_r_joint",
            31: "body_yaw_joint",
            51: "hip_roll_l_joint",  52: "hip_pitch_l_joint",  53: "hip_yaw_l_joint",
            54: "knee_pitch_l_joint", 55: "ankle_pitch_l_joint", 56: "ankle_roll_l_joint",
            61: "hip_roll_r_joint",  62: "hip_pitch_r_joint",  63: "hip_yaw_r_joint",
            64: "knee_pitch_r_joint", 65: "ankle_pitch_r_joint", 66: "ankle_roll_r_joint",
        }
        self.NAME_TO_ID = {v: k for k, v in self.ID_TO_NAME.items()}

        self.HAND_L_MAP = {
            1: "left_little_1_joint", 2: "left_ring_1_joint",
            3: "left_middle_1_joint", 4: "left_index_1_joint",
            5: "left_thumb_2_joint",  6: "left_thumb_1_joint",
        }
        self.HAND_R_MAP = {
            1: "right_little_1_joint", 2: "right_ring_1_joint",
            3: "right_middle_1_joint", 4: "right_index_1_joint",
            5: "right_thumb_2_joint",  6: "right_thumb_1_joint",
        }

        self.create_subscription(CmdSetMotorPosition, "/arm/cmd_pos",   self.cmd_pos_cb, 10)
        self.create_subscription(CmdSetMotorPosition, "/head/cmd_pos",  self.cmd_pos_cb, 10)
        self.create_subscription(CmdSetMotorPosition, "/leg/cmd_pos",   self.cmd_pos_cb, 10)
        self.create_subscription(CmdSetMotorPosition, "/waist/cmd_pos", self.cmd_pos_cb, 10)
        self.create_subscription(
            JointState, "/inspire_hand/ctrl/left_hand",
            lambda m: self.hand_cb(m, "left"), 10,
        )
        self.create_subscription(
            JointState, "/inspire_hand/ctrl/right_hand",
            lambda m: self.hand_cb(m, "right"), 10,
        )

        self.pub_arm_status   = self.create_publisher(MotorStatusMsg, "/arm/status", 10)
        self.pub_head_status  = self.create_publisher(MotorStatusMsg, "/head/status", 10)
        self.pub_leg_status   = self.create_publisher(MotorStatusMsg, "/leg/status", 10)
        self.pub_waist_status = self.create_publisher(MotorStatusMsg, "/waist/status", 10)
        self.pub_hand_l_status = self.create_publisher(JointState, "/inspire_hand/state/left_hand", 10)
        self.pub_hand_r_status = self.create_publisher(JointState, "/inspire_hand/state/right_hand", 10)

        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        self.get_logger().info("[BRIDGE] Tiangong ROS2 Bridge (v2 - sim closed-loop) started")

    def cmd_pos_cb(self, msg: CmdSetMotorPosition):
        """Forward arm/head/leg/waist motor commands from ROS2 to Isaac Lab."""
        full_action = {}
        for jname in self.ID_TO_NAME.values():
            full_action[jname] = 0.0

        valid = 0
        for cmd in msg.cmds:
            jname = self.ID_TO_NAME.get(cmd.name)
            if jname:
                full_action[jname] = float(cmd.pos)
                valid += 1
                if self._debug_cmd:
                    print(f"[BRIDGE][ARM] id={cmd.name} {jname} -> {cmd.pos:.6f}", flush=True)
            elif self._debug_cmd:
                print(f"[BRIDGE][WARN] unknown motor id={cmd.name} pos={cmd.pos:.6f}", flush=True)

        if full_action:
            self.cmd_socket.send_json(full_action)
            if self._debug_cmd:
                print(
                    f"[BRIDGE][ZMQ SEND ARM] ros_cmds={len(msg.cmds)} "
                    f"valid={valid} zmq_joints={len(full_action)}",
                    flush=True,
                )

    def hand_cb(self, msg: JointState, side: str):
        """Forward hand control commands from ROS2 to Isaac Lab."""
        action = {}
        map_target = self.HAND_L_MAP if side == "left" else self.HAND_R_MAP
        for i, name in enumerate(msg.name):
            try:
                id_val = int(name)
                jname = map_target.get(id_val)
                if jname:
                    action[jname] = float(msg.position[i])
            except (ValueError, IndexError):
                pass
        if action:
            self.cmd_socket.send_json(action)
            if self._debug_cmd:
                print(
                    f"[BRIDGE][ZMQ SEND HAND] side={side} joints={len(action)} "
                    f"keys={list(action.keys())}",
                    flush=True,
                )

    def _poll_loop(self):
        """Background thread: poll ZMQ status socket and publish to ROS2."""
        poller = zmq.Poller()
        poller.register(self.status_socket, zmq.POLLIN)
        while self._running and rclpy.ok():
            socks = dict(poller.poll(timeout=1))
            if self.status_socket in socks:
                try:
                    msg = self.status_socket.recv_json(flags=zmq.NOBLOCK)
                    self._publish_status(msg)
                except zmq.Again:
                    pass
                except Exception:
                    pass

    def _publish_status(self, data: dict):
        """Convert Isaac Lab status payload into individual ROS2 status topics."""
        header = self.get_clock().now().to_msg()

        if isinstance(data, dict) and "joint_names" in data:
            pos_map = dict(zip(data["joint_names"], data["joint_pos"]))
            finger_pct = data.get("finger_percentages", {})
            vel_map = (
                dict(zip(data["joint_names"], data["joint_vel"]))
                if "joint_vel" in data
                else {}
            )
        else:
            pos_map = data
            finger_pct = {}
            vel_map = {}

        def _make_motor_msg(id_range):
            m = MotorStatusMsg()
            m.header.stamp = header
            for name, mid in self.NAME_TO_ID.items():
                if mid in id_range and name in pos_map:
                    s = MotorStatus()
                    s.name = mid
                    s.pos = float(pos_map[name])
                    m.status.append(s)
            return m

        def _make_hand_msg(hand_map):
            m = JointState()
            m.header.stamp = header
            m.name = [str(i) for i in range(1, 7)]
            m.position = []
            m.velocity = []
            for i in range(1, 7):
                sim_name = hand_map.get(i)
                if sim_name and sim_name in finger_pct:
                    m.position.append(float(finger_pct[sim_name]))
                    m.velocity.append(float(vel_map.get(sim_name, 0.0)))
                else:
                    m.position.append(0.0)
                    m.velocity.append(0.0)
                m.effort.append(0.0)
            return m

        self.pub_arm_status.publish(_make_motor_msg(self.ID_ARM_L + self.ID_ARM_R))
        self.pub_head_status.publish(_make_motor_msg(self.ID_HEAD))
        self.pub_leg_status.publish(_make_motor_msg(self.ID_LEG_L + self.ID_LEG_R))
        self.pub_waist_status.publish(_make_motor_msg(self.ID_WAIST))
        self.pub_hand_l_status.publish(_make_hand_msg(self.HAND_L_MAP))
        self.pub_hand_r_status.publish(_make_hand_msg(self.HAND_R_MAP))

    def stop(self):
        """Clean shutdown of ZMQ sockets and background threads."""
        self._running = False
        self._poll_thread.join(timeout=2.0)
        self._img_receiver.stop()
        self.status_socket.close()
        self.cmd_socket.close()
        self._zmq_context.term()

    def destroy_node(self):
        self.stop()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="Tiangong ROS2 ZMQ Bridge")
    parser.add_argument(
        "--sim-host", type=str, default="127.0.0.1",
        help="IP or hostname of the machine running Isaac Lab. "
        "Use 127.0.0.1 with Docker --network host.",
    )
    parser.add_argument(
        "--cmd-port", type=int, default=6555,
        help="ZMQ port for ROS2 commands -> Isaac.",
    )
    parser.add_argument(
        "--status-port", type=int, default=6556,
        help="ZMQ port for Isaac status -> ROS2.",
    )
    parser.add_argument(
        "--image-port", type=int, default=6557,
        help="ZMQ port for Isaac images -> ROS2.",
    )
    parser.add_argument(
        "--debug-cmd", action="store_true",
        help="Print ROS command reception and ZMQ send details.",
    )
    args, _ = parser.parse_known_args()

    rclpy.init(args=sys.argv)
    node = TiangongRosBridge(
        sim_host=args.sim_host,
        cmd_port=args.cmd_port,
        status_port=args.status_port,
        image_port=args.image_port,
        debug_cmd=args.debug_cmd,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
