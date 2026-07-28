from typing import Any
import time
import torch
import json
import numpy as np
import cv2
from ..device_base import DeviceBase
from .tiangong_pro_action_process import to_controller_data, to_ros_data

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

class TiangongProController(DeviceBase):
    """Controller for Tiangong Pro that receives actions via ZMQ bridge.
    This allows ROS 2 Humble (Python 3.10) to communicate with this environment (Python 3.11).
    """
    def __init__(self, env, **kwargs):
        super().__init__()
        self.env = env
        self.device = env.device
        self._last_camera_send_time = -1.0
        
        self._last_depth_send_time = -1.0
        # Initialize default action buffer
        self._action = {}

        if HAS_ZMQ:
            self.context = zmq.Context()
            # Subscriber for control commands
            self.sub_socket = self.context.socket(zmq.SUB)
            self.sub_socket.connect("tcp://127.0.0.1:5555")
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            
            # Publisher for status feedback
            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.setsockopt(zmq.SNDHWM, 1)
            self.pub_socket.bind("tcp://*:5556")

            # Publisher for camera images
            self.img_socket = self.context.socket(zmq.PUB)
            self.img_socket.setsockopt(zmq.SNDHWM, 1) # Only keep latest image
            self.img_socket.bind("tcp://*:5557")
            
            print("[INFO] ZMQ Subscriber connected to tcp://127.0.0.1:5555")
            print("[INFO] ZMQ Publisher bound to tcp://*:5556")
            print("[INFO] ZMQ Image Publisher bound to tcp://*:5557")
        else:
            print("[WARNING] zmq not found. ZMQ control will not be available.")

    def __str__(self) -> str:
        return "Tiangong Pro ZMQ Controller"

    def reset(self):
        self._action = {}

    def reset_to_pose(self, pose_dict: dict[str, float]):
        """Reset the internal action buffer to a specific pose."""
        self._action = pose_dict.copy()

    def add_callback(self, key, func):
        pass

    def _get_status_from_sim(self) -> dict:
        """Fetch current robot state from simulation."""
        
        return to_ros_data(self.env)

    def _send_camera_data(self):
        """Fetch and send camera data over ZMQ."""
        if "camera" not in self.env.scene.keys():
            return
        

        camera = self.env.scene["camera"]
        camera_depth = self.env.scene["camera_depth"]
        try:
            rgb_tensor = camera.data.output["rgb"]
            depth_tensor = camera_depth.data.output.get("depth") if camera_depth.data.output is not None else None
            if rgb_tensor is None or rgb_tensor.shape[0] == 0:
                return

            # Pull tensors to CPU and convert to bytes
            rgb = rgb_tensor[0].cpu().numpy()

            depth_bytes = b""
            if depth_tensor is not None and depth_tensor.shape[0] != 0:
                # Convert to mm (x1000) and uint16 for standard ROS depth
                depth_img_meters = depth_tensor[0].cpu().numpy()
                depth_img_mm = depth_img_meters * 1000.0
                # Clip to safe range for uint16
                depth_img_mm = np.clip(depth_img_mm, 0, 65535)
                depth_img = depth_img_mm.astype(np.uint16)
                depth_bytes = depth_img.tobytes()

            # send raw bytes (preserve full quality)
            encoded_rgb = rgb.tobytes()

            metadata = {
                "width": rgb.shape[1],
                "height": rgb.shape[0],
                "format": "raw", # Raw bytes format
            }

            # Send as multi-part message (raw bytes for maximum quality)
            self.img_socket.send_json(metadata, flags=zmq.SNDMORE | zmq.NOBLOCK)
            self.img_socket.send(encoded_rgb, flags=zmq.SNDMORE | zmq.NOBLOCK)
            self.img_socket.send(depth_bytes, flags=zmq.NOBLOCK)

            return
        except Exception:
            return

    def advance(self) -> dict[str, Any]:
        """
        Receive actions from ZMQ and send status back.
        """
        if HAS_ZMQ:
            # 1. Receive control commands
            while True:
                try:
                    msg = self.sub_socket.recv_json(flags=zmq.NOBLOCK)
                    self._action.update(msg)
                except zmq.Again:
                    break
            try:
                status = self._get_status_from_sim()
                self.pub_socket.send_json(status, flags=zmq.NOBLOCK)
            except Exception:
                pass
            try:
                # send camera data (ignore timing return)
                self._send_camera_data()
            except Exception:
                pass

        return {"tiangong_pro": to_controller_data(self._action,self.env)}

    def display_controls(self):
        """Display the controls."""
        if HAS_ZMQ:
            print("Tiangong Pro Controller: Full ROS Interface enabled via ZMQ Bridge")
            print("  - Command Sub: tcp://127.0.0.1:5555")
            print("  - Status Pub:  tcp://127.0.0.1:5556")
