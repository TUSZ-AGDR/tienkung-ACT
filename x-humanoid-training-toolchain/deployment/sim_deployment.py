#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulation Closed-Loop Deployment for Tiangong Pro + ACT Policy.

Replaces real-camera/hardware pipeline with simulation topics.

Architecture:
  [ACT Policy] <- /sim/camera_head/color/image_raw  (from ZMQ bridge)
  [ACT Policy] <- /arm/status, /inspire_hand/state/* (from ZMQ bridge)
  [ACT Policy] -> /arm/cmd_pos, /inspire_hand/ctrl/* (to ZMQ bridge)

26-Dim Action Space (robomind format):
  action[0:7]   = left_arm  (7 joints)  -> motor IDs 11-17
  action[7:13]  = left_hand (6 DOF, 0-1 percentage) -> /inspire_hand/ctrl/left_hand
  action[13:20] = right_arm (7 joints)  -> motor IDs 21-27
  action[20:26] = right_hand (6 DOF, 0-1 percentage) -> /inspire_hand/ctrl/right_hand

Usage:
  python sim_deployment.py --model-path /path/to/act_model
"""

import argparse
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

try:
    from bodyctrl_msgs.msg import CmdSetMotorPosition, MotorStatusMsg, SetMotorPosition
except ImportError:
    print("[ERROR] bodyctrl_msgs not found. Please source the correct workspace.")
    sys.exit(1)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from action_policy import PolicyAgent


class SimDeploymentNode(Node):
    """Policy inference node for simulation closed-loop control."""

    def __init__(self, model_path: str, print_hand_every: int = 1):
        super().__init__("sim_deployment_node")
        self.print_hand_every = max(1, int(print_hand_every))

        # ---- Policy ----
        self.get_logger().info(f"Loading policy from: {model_path}")
        self.action_policy = PolicyAgent(model_path)
        self.get_logger().info("Policy loaded successfully")

        # ---- Arm command publisher ----
        self.arm_cmd_pub = self.create_publisher(CmdSetMotorPosition, "/arm/cmd_pos", 10)

        # ---- Hand command publishers ----
        self.left_hand_pub = self.create_publisher(JointState, "/inspire_hand/ctrl/left_hand", 10)
        self.right_hand_pub = self.create_publisher(JointState, "/inspire_hand/ctrl/right_hand", 10)

        # ---- Subscriptions (from ZMQ bridge) ----
        self._latest_rgb = None
        self.create_subscription(
            Image, "/sim/camera_head/color/image_raw", self._image_cb, 10
        )

        self._left_jpos = None
        self._right_jpos = None
        self.create_subscription(
            MotorStatusMsg, "/arm/status", self._arm_status_cb, 10
        )

        self._left_hand_pos = np.zeros(6, dtype=np.float32)
        self._right_hand_pos = np.zeros(6, dtype=np.float32)
        self.create_subscription(
            JointState, "/inspire_hand/state/left_hand", self._hand_l_cb, 10
        )
        self.create_subscription(
            JointState, "/inspire_hand/state/right_hand", self._hand_r_cb, 10
        )

        # ---- Inference control ----
        self._inference_lock = threading.Lock()
        self._step_count = 0

        self.get_logger().info("SimDeploymentNode initialized - ready for simulation closed-loop")

    # -------------------------------------------------------------------
    # Callbacks - cache latest sensor data
    # -------------------------------------------------------------------
    def _image_cb(self, msg: Image):
        """Parse raw RGB bytes from simulation camera image topic."""
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            self._latest_rgb = data.reshape((msg.height, msg.width, 3))
        except Exception as e:
            self.get_logger().error(f"Manual image parsing failed: {e}")

    def _arm_status_cb(self, msg: MotorStatusMsg):
        """Parse arm MotorStatusMsg into left_arm(7) and right_arm(7) arrays.

        Motor IDs: left arm 11-17, right arm 21-27.
        """
        pos_map = {}
        for s in msg.status:
            pos_map[s.name] = float(s.pos)
        self._left_jpos = np.array([pos_map.get(i, 0.0) for i in range(11, 18)], dtype=np.float32)
        self._right_jpos = np.array([pos_map.get(i, 0.0) for i in range(21, 28)], dtype=np.float32)

    def _hand_l_cb(self, msg: JointState):
        if len(msg.position) >= 6:
            self._left_hand_pos = np.array(msg.position[:6], dtype=np.float32)

    def _hand_r_cb(self, msg: JointState):
        if len(msg.position) >= 6:
            self._right_hand_pos = np.array(msg.position[:6], dtype=np.float32)

    # -------------------------------------------------------------------
    # Observation assembly (matches ACT training data format)
    # -------------------------------------------------------------------
    def _get_observation(self) -> dict | None:
        """Build observation dict in the format expected by PolicyAgent.

        Returns:
            dict | None: Observation dict with 'images' and 'arm_gripper_joints',
            or None if the first image frame has not been received yet.
        """
        rgb = self._latest_rgb
        if rgb is None:
            return None

        # Build 26-dim proprioceptive state:
        # [left_arm(7), left_hand(6), right_arm(7), right_hand(6)]
        left_arm = self._left_jpos if self._left_jpos is not None else np.zeros(7)
        right_arm = self._right_jpos if self._right_jpos is not None else np.zeros(7)
        proprio = np.concatenate([left_arm, self._left_hand_pos, right_arm, self._right_hand_pos])

        return {
            "images": {"camera_head": rgb},
            "arm_gripper_joints": proprio,
        }

    # -------------------------------------------------------------------
    # Action construction and publishing
    # -------------------------------------------------------------------
    def _construct_arm_cmd(self, target_joint: np.ndarray) -> CmdSetMotorPosition:
        """Build CmdSetMotorPosition for dual arms (14 joints).

        Args:
            target_joint: Array of shape (14,) with [left_arm(7), right_arm(7)].
                          Motor IDs: left 11-17, right 21-27.

        Returns:
            CmdSetMotorPosition: Filled ROS message ready for publishing.
        """
        msg = CmdSetMotorPosition()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        for idx, val in enumerate(target_joint):
            cmd = SetMotorPosition()
            cmd.name = (11 + idx) if idx < 7 else (14 + idx)
            cmd.pos = float(val)
            cmd.spd = 0.5
            cmd.cur = 5.0
            msg.cmds.append(cmd)
        return msg

    def _construct_hand_cmd(self, positions: np.ndarray) -> JointState:
        """Build JointState for Inspire hand (6 DOF, 0-1 percentage)."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["1", "2", "3", "4", "5", "6"]
        msg.position = [float(np.clip(p, 0.0, 1.0)) for p in positions]
        return msg

    def publish_action(self, action: np.ndarray):
        """Publish the 26-dim action to ROS2 control topics.

        Action layout:
          [0:7]   = left_arm  -> /arm/cmd_pos (IDs 11-17)
          [7:13]  = left_hand -> /inspire_hand/ctrl/left_hand
          [13:20] = right_arm -> /arm/cmd_pos (IDs 21-27)
          [20:26] = right_hand -> /inspire_hand/ctrl/right_hand

        Args:
            action: Array of shape (26,) following the ACT action convention.
        """
        target_joint = np.concatenate([action[:7], action[13:20]])
        left_hand_pos = action[7:13]
        right_hand_pos = action[20:26]

        self.arm_cmd_pub.publish(self._construct_arm_cmd(target_joint))
        self.left_hand_pub.publish(self._construct_hand_cmd(left_hand_pos))
        self.right_hand_pub.publish(self._construct_hand_cmd(right_hand_pos))

    @staticmethod
    def _fmt_vec(values: np.ndarray) -> str:
        """Format a numeric vector for readable debug logging."""
        return ", ".join(f"{float(v):.4f}" for v in values)

    def _print_inferred_hand_action(self, action: np.ndarray):
        """Print the 12-D hand action inferred by the policy, split left/right.

        Action layout follows the 26-D robomind convention:
          action[7:13]  = left hand 6-D percentage command
          action[20:26] = right hand 6-D percentage command
        These are the raw policy outputs before clipping in _construct_hand_cmd().

        Args:
            action: Array of shape (26,) containing the raw policy output.
        """
        left_hand_raw = action[7:13]
        right_hand_raw = action[20:26]
        right_hand_clip = np.clip(right_hand_raw, 0.0, 1.0)

        self.get_logger().info(
            f"Step {self._step_count} | Inferred hand action raw 12D | "
            f"L_hand[6]: [{self._fmt_vec(left_hand_raw)}] | "
            f"R_hand[6]: [{self._fmt_vec(right_hand_raw)}]"
        )
        right_names = [
            "right_little_1_joint",
            "right_ring_1_joint",
            "right_middle_1_joint",
            "right_index_1_joint",
            "right_thumb_2_joint",
            "right_thumb_1_joint",
        ]
        self.get_logger().info(
            "PUBLISH right_hand named | "
            + " | ".join(f"{n}={v:.6f}" for n, v in zip(right_names, right_hand_clip))
        )

    # -------------------------------------------------------------------
    # Main inference loop
    # -------------------------------------------------------------------
    def run_loop(self):
        """Run policy inference in a loop, triggered by incoming images."""
        self.get_logger().info("Starting simulation closed-loop inference...")
        self.get_logger().info("Waiting for first image frame...")

        while rclpy.ok() and self._latest_rgb is None:
            time.sleep(0.1)
        self.get_logger().info("First image received, starting policy inference")

        self.action_policy.reset()

        while rclpy.ok():
            obs = self._get_observation()
            if obs is None:
                time.sleep(0.01)
                continue

            try:
                with self._inference_lock:
                    action = self.action_policy.inference(obs)
            except Exception as e:
                self.get_logger().error(f"Inference error: {e}")
                time.sleep(0.1)
                continue

            action_np = action[0].detach().cpu().numpy()

            self._step_count += 1
            if self._step_count % self.print_hand_every == 0:
                self._print_inferred_hand_action(action_np)

            self.publish_action(action_np)

            if self._step_count % 100 == 0:
                self.get_logger().info(
                    f"Step {self._step_count} | "
                    f"L_arm: [{', '.join(f'{v:.3f}' for v in action_np[:7])}] | "
                    f"R_arm: [{', '.join(f'{v:.3f}' for v in action_np[13:20])}]"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simulation Closed-Loop Policy Deployment")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the ACT policy checkpoint directory",
    )
    parser.add_argument(
        "--print-hand-every", type=int, default=1,
        help="Print inferred left/right 6-D hand actions every N inference steps. Default: 1.",
    )
    args = parser.parse_args()

    rclpy.init(args=sys.argv)

    try:
        node = SimDeploymentNode(model_path=args.model_path, print_hand_every=args.print_hand_every)

        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)

        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()

        node.run_loop()

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
