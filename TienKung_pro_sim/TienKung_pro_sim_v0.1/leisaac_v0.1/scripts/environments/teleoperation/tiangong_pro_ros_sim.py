# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tiangong Pro Isaac-Lab ZMQ simulation server for ROS2 closed-loop ACT deployment.

This script is the Isaac-side counterpart of ros2_zmq_bridge2.py.

Network protocol:
  ROS2/ZMQ bridge  --PUB connect-->  tcp://<sim-host>:6555  --SUB bind-->  this script
  ROS2/ZMQ bridge  <--SUB connect--  tcp://<sim-host>:6556  <--PUB bind--  this script
  ROS2/ZMQ bridge  <--SUB connect--  tcp://<sim-host>:6557  <--PUB bind--  this script

It does three things:
  1) Receives joint-name -> target-position JSON commands from ZMQ port (--cmd_port, default 6555).
  2) Applies those commands to the Isaac robot joint position targets.
  3) Publishes robot joint status on --status_port (default 6556) and RGB images on --image_port (default 6557).

Typical launch order:
  Terminal A, Isaac side:
    ./isaaclab.sh -p tiangong_pro_zmq_sim.py --task <YOUR_TASK> --device cuda --enable_cameras --cmd_port 6555 --status_port 6556 --image_port 6557

  Terminal B, ROS2 side:
    source <your_ros2_ws>/install/setup.bash
    python ros2_zmq_bridge2.py --sim-host <ISAAC_MACHINE_IP>

  Terminal C, ROS2 side:
    source <your_ros2_ws>/install/setup.bash
    python sim_deployment.py --model-path <ACT_MODEL_DIR>
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import signal
import sys
import time
from typing import Dict, Iterable, Optional, Tuple

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tiangong Pro Isaac-Lab ZMQ simulation server.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate. Use 1 for ROS closed loop.")
parser.add_argument("--task", type=str, default=None, help="Name of the Isaac-Lab task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--step_hz", type=int, default=60, help="Simulation stepping rate in Hz.")
parser.add_argument("--cmd_port", type=int, default=6555, help="ZMQ SUB port for incoming commands. Default avoids existing service on 5555.")
parser.add_argument("--status_port", type=int, default=6556, help="ZMQ PUB port for outgoing robot status.")
parser.add_argument("--image_port", type=int, default=6557, help="ZMQ PUB port for outgoing images.")
parser.add_argument("--image_hz", type=float, default=20.0, help="RGB image publishing rate.")
parser.add_argument("--status_hz", type=float, default=60.0, help="Robot status publishing rate.")
parser.add_argument("--head_pitch", type=float, default=+0.25, help="Fixed head pitch target. Tune the sign/value if the robot raises its head instead of lowering it.")
parser.add_argument(
    "--thumb1_from_thumb2", "--thumb1-from-thumb2",
    dest="thumb1_from_thumb2",
    action="store_true",
    default=True,
    help=(
        "If thumb_1 command is missing or near zero, copy thumb_2 percentage to thumb_1. "
        "This fixes the thumb root staying still while keeping other fingers master-joint only."
    ),
)
parser.add_argument(
    "--no_thumb1_from_thumb2", "--no-thumb1-from-thumb2",
    dest="thumb1_from_thumb2",
    action="store_false",
    help="Disable thumb_2 -> thumb_1 coupling and use the incoming 6th thumb channel exactly as received.",
)
parser.add_argument(
    "--thumb1_coupling_scale", "--thumb1-coupling-scale",
    type=float,
    default=1.0,
    help="Scale used when copying thumb_2 percentage to thumb_1. Only used when --thumb1-from-thumb2 is enabled.",
)
parser.add_argument("--no_image", action="store_true", help="Disable image publishing.")
parser.add_argument("--debug_cmd", action="store_true", help="Print received command and target-application details.")
parser.add_argument("--debug_cmd_every", type=int, default=10, help="When --debug_cmd is enabled, print one detailed command log every N received commands.")
parser.add_argument("--debug_joint_delta", type=float, default=1e-4, help="Delta threshold for debug movement checks.")
parser.add_argument("--hand_input_mode", type=str, default="pct_close1", choices=["pct_close1", "pct_open1", "qpos"], help="How to interpret incoming hand commands. pct_close1: 1 means close and maps to joint limit; pct_open1: 1 means open using official tiangong_pro_action_process formula; qpos: incoming value is raw joint qpos.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Reduce Isaac/Omniverse log noise.
sys.argv.append("--/log/level=error")
sys.argv.append("--/log/fileLogLevel=error")
sys.argv.append("--/log/outputStreamLevel=error")

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import zmq
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.assets.robots.tiangong_pro import (
    TIANGONG_PRO_LEFT_ARM_JOINTS,
    TIANGONG_PRO_RIGHT_ARM_JOINTS,
    TIANGONG_PRO_LEFT_HAND_JOINTS,
    TIANGONG_PRO_RIGHT_HAND_JOINTS,
    TIANGONG_PRO_JOINT_LIMITS,
    TIANGONG_PRO_MIMIC_JOINTS,
)

try:
    from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
except Exception:
    dynamic_reset_gripper_effort_limit_sim = None


# -----------------------------------------------------------------------------
# Joint-name definitions. These names must match ros2_zmq_bridge2.py.
# -----------------------------------------------------------------------------
ID_TO_NAME = {
    1: "head_roll_joint",  2: "head_pitch_joint",  3: "head_yaw_joint",
    11: "shoulder_pitch_l_joint", 12: "shoulder_roll_l_joint", 13: "shoulder_yaw_l_joint",
    14: "elbow_pitch_l_joint",    15: "elbow_yaw_l_joint",
    16: "wrist_pitch_l_joint",    17: "wrist_roll_l_joint",
    21: "shoulder_pitch_r_joint", 22: "shoulder_roll_r_joint", 23: "shoulder_yaw_r_joint",
    24: "elbow_pitch_r_joint",    25: "elbow_yaw_r_joint",
    26: "wrist_pitch_r_joint",    27: "wrist_roll_r_joint",
    31: "body_yaw_joint",
    51: "hip_roll_l_joint",  52: "hip_pitch_l_joint",  53: "hip_yaw_l_joint",
    54: "knee_pitch_l_joint", 55: "ankle_pitch_l_joint", 56: "ankle_roll_l_joint",
    61: "hip_roll_r_joint",  62: "hip_pitch_r_joint",  63: "hip_yaw_r_joint",
    64: "knee_pitch_r_joint", 65: "ankle_pitch_r_joint", 66: "ankle_roll_r_joint",
}

HAND_L_MAP = {
    1: "left_little_1_joint", 2: "left_ring_1_joint",
    3: "left_middle_1_joint", 4: "left_index_1_joint",
    5: "left_thumb_2_joint",  6: "left_thumb_1_joint",
}
HAND_R_MAP = {
    1: "right_little_1_joint", 2: "right_ring_1_joint",
    3: "right_middle_1_joint", 4: "right_index_1_joint",
    5: "right_thumb_2_joint",  6: "right_thumb_1_joint",
}

ALL_BRIDGE_JOINT_NAMES = list(ID_TO_NAME.values()) + list(HAND_L_MAP.values()) + list(HAND_R_MAP.values())


# -----------------------------------------------------------------------------
# Tiangong hand command conversion.
#
# The original tiangong_pro_action_process.py uses:
#       qpos = (1.0 - percentage) * target_limit
# which means percentage=1 is open. In this deployment, the policy/ROS hand
# command logs show cmd≈1 should be the target close qpos, so the default mode is
# pct_close1:
#       qpos = percentage * target_limit
#
# Mimic joints still follow TIANGONG_PRO_MIMIC_JOINTS. Arm initialization and
# arm direct targets stay exactly as in the original script.
# -----------------------------------------------------------------------------
HAND_MASTER_NAMES = set(HAND_L_MAP.values()) | set(HAND_R_MAP.values())
HAND_ALL_NAMES = set(TIANGONG_PRO_LEFT_HAND_JOINTS) | set(TIANGONG_PRO_RIGHT_HAND_JOINTS) | HAND_MASTER_NAMES


def _is_hand_joint_name(name: str) -> bool:
    return name in HAND_ALL_NAMES or name.startswith("left_") or name.startswith("right_")


def _hand_cmd_to_joint_target(joint_name: str, value: float, mode: str) -> float:
    """Convert incoming hand command to raw Isaac joint qpos target.

    mode choices:
      - pct_close1: incoming 1.0 means close. qpos = value * target_limit.
        This matches the current policy/ROS command convention shown by the logs.
      - pct_open1: incoming 1.0 means open. qpos = (1 - value) * target_limit.
        This is the formula in tiangong_pro_action_process.py.
      - qpos: incoming value is already raw qpos.
    """
    v = float(value)
    if mode == "qpos":
        return v

    if joint_name in TIANGONG_PRO_JOINT_LIMITS:
        low, high, *_ = TIANGONG_PRO_JOINT_LIMITS[joint_name]
        target_limit = low if abs(low) > abs(high) else high
        if mode == "pct_open1":
            return float(1.0 - v) * float(target_limit)
        # Default for this deployment: 1.0 means close.
        return float(v) * float(target_limit)

    return v


def _apply_mimic_targets_for_source(
    target: torch.Tensor,
    joint_map: Dict[str, int],
    source_joint: str,
    source_target: float,
    visited: Optional[set[str]] = None,
):
    """Recursively apply all mimic joints driven by source_joint.

    The official tiangong_pro_action_process.py computes mimic values recursively:
        mimic = multiplier * get_val(source_joint)
    Directly setting only the six master joints is visually weak because many
    visible phalanx joints are mimic/dependent joints. This function expands the
    whole mimic tree every time a master finger command is received.
    """
    if visited is None:
        visited = set()
    if source_joint in visited:
        return
    visited.add(source_joint)

    for mimic_name, mimic_info in TIANGONG_PRO_MIMIC_JOINTS.items():
        if mimic_info.get("joint") != source_joint:
            continue
        multiplier = float(mimic_info.get("multiplier", 1.0))
        mimic_target = multiplier * float(source_target)
        jid = joint_map.get(mimic_name)
        if jid is not None:
            target[:, jid] = mimic_target
        # Some mimic joints may themselves be sources of further mimic joints.
        _apply_mimic_targets_for_source(target, joint_map, mimic_name, mimic_target, visited)


def _refresh_all_hand_mimic_targets(target: torch.Tensor, joint_map: Dict[str, int]):
    """Make all hand mimic targets consistent with their current source targets.

    This is deliberately run after each command because ROS messages for hands
    contain only six master channels, while the USD/action model can contain many
    additional finger joints.
    """
    # Iterate several times to cover nested mimic chains even if the dictionary
    # order is not source-before-child.
    for _ in range(4):
        changed = False
        for mimic_name, mimic_info in TIANGONG_PRO_MIMIC_JOINTS.items():
            mid = joint_map.get(mimic_name)
            source = mimic_info.get("joint")
            sid = joint_map.get(source)
            if mid is None or sid is None:
                continue
            multiplier = float(mimic_info.get("multiplier", 1.0))
            new_val = target[:, sid] * multiplier
            if torch.max(torch.abs(target[:, mid] - new_val)).item() > 1e-7:
                target[:, mid] = new_val
                changed = True
        if not changed:
            break


def _finger_percentages_from_robot(robot, mode: str) -> Dict[str, float]:
    """Inverse conversion for hand state feedback, matching hand_input_mode."""
    joint_names = list(robot.data.joint_names)
    joint_pos = robot.data.joint_pos[0].detach().cpu().numpy().tolist()
    joint_pos_map = dict(zip(joint_names, joint_pos))
    finger_percentages: Dict[str, float] = {}

    for name, limits in TIANGONG_PRO_JOINT_LIMITS.items():
        if name in TIANGONG_PRO_MIMIC_JOINTS:
            continue
        if name not in joint_pos_map:
            continue
        current_val = float(joint_pos_map[name])
        low, high = limits[0], limits[1]
        target_limit = low if abs(low) > abs(high) else high
        if abs(float(target_limit)) > 1e-6:
            if mode == "pct_open1":
                percentage = 1.0 - current_val / float(target_limit)
            elif mode == "pct_close1":
                percentage = current_val / float(target_limit)
            else:
                # In raw-qpos mode the policy should not treat this as a percentage;
                # publish a clipped normalized value for compatibility.
                percentage = current_val / float(target_limit)
            finger_percentages[name] = float(np.clip(percentage, 0.0, 1.0))
        else:
            finger_percentages[name] = 1.0 if mode == "pct_open1" else 0.0
    return finger_percentages


class RateLimiter:
    def __init__(self, hz: float):
        self.hz = float(hz)
        self.last_time = time.perf_counter()
        self.sleep_duration = 1.0 / max(self.hz, 1e-6)

    def update_from_env(self, env):
        try:
            sim_dt = float(env.cfg.sim.dt)
            dec = int(getattr(env.cfg, "decimation", 1))
            self.sleep_duration = max(sim_dt * max(1, dec), 1.0 / max(self.hz, 1e-6))
        except Exception:
            self.sleep_duration = 1.0 / max(self.hz, 1e-6)

    def sleep(self):
        now = time.perf_counter()
        to_sleep = self.sleep_duration - (now - self.last_time)
        if to_sleep > 0:
            time.sleep(to_sleep)
            self.last_time += self.sleep_duration
        else:
            self.last_time = now


class IsaacZmqServer:
    """ZMQ server running inside the Isaac process."""

    def __init__(self, cmd_port: int, status_port: int, image_port: int):
        self.context = zmq.Context.instance()

        self.cmd_sub = self.context.socket(zmq.SUB)
        self.cmd_sub.bind(f"tcp://*:{cmd_port}")
        self.cmd_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.cmd_sub.setsockopt(zmq.RCVHWM, 1)
        self.cmd_sub.setsockopt(zmq.CONFLATE, 1)

        self.status_pub = self.context.socket(zmq.PUB)
        self.status_pub.bind(f"tcp://*:{status_port}")
        self.status_pub.setsockopt(zmq.SNDHWM, 2)

        self.image_pub = self.context.socket(zmq.PUB)
        self.image_pub.bind(f"tcp://*:{image_port}")
        self.image_pub.setsockopt(zmq.SNDHWM, 2)

        print(f"[ZMQ] CMD    SUB bind tcp://*:{cmd_port}")
        print(f"[ZMQ] STATUS PUB bind tcp://*:{status_port}")
        print(f"[ZMQ] IMAGE  PUB bind tcp://*:{image_port}")
        time.sleep(0.5)  # Give PUB/SUB sockets a moment to connect.

    def recv_latest_cmd(self) -> Optional[Dict[str, float]]:
        latest = None
        while True:
            try:
                msg = self.cmd_sub.recv_json(flags=zmq.NOBLOCK)
                if isinstance(msg, dict):
                    latest = {str(k): float(v) for k, v in msg.items()}
            except zmq.Again:
                break
            except Exception as e:
                print(f"[ZMQ][WARN] Failed to receive command: {e}")
                break
        return latest

    def publish_status(self, payload: Dict):
        try:
            self.status_pub.send_json(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception as e:
            print(f"[ZMQ][WARN] Failed to publish status: {e}")

    def publish_rgb(self, rgb: np.ndarray):
        if rgb is None:
            return
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return
        rgb = np.ascontiguousarray(rgb[..., :3])
        h, w = rgb.shape[:2]
        meta = {"width": int(w), "height": int(h), "format": "raw", "encoding": "rgb8"}
        try:
            self.image_pub.send_json(meta, flags=zmq.SNDMORE | zmq.NOBLOCK)
            self.image_pub.send(rgb.tobytes(), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception as e:
            print(f"[ZMQ][WARN] Failed to publish image: {e}")

    def close(self):
        self.cmd_sub.close(0)
        self.status_pub.close(0)
        self.image_pub.close(0)
        # Do not term Context.instance(); Isaac/other libs may share it.


def _as_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _first_env_vector(x) -> np.ndarray:
    arr = _as_np(x)
    if arr.ndim >= 2:
        return arr[0]
    return arr


def _get_robot(env):
    if "robot" in env.scene.articulations:
        return env.scene["robot"]
    # Fallback: first articulation.
    keys = list(env.scene.articulations.keys())
    if not keys:
        raise RuntimeError("No articulation found in env.scene.articulations.")
    print(f"[WARN] 'robot' articulation not found. Using first articulation: {keys[0]}")
    return env.scene[keys[0]]


def _resolve_joint_map(robot, candidate_names: Iterable[str]) -> Dict[str, int]:
    """Return a map of joint_name -> joint_id for names that exist in the robot."""
    out = {}
    for name in candidate_names:
        try:
            ids, names = robot.find_joints([name])
            if len(ids) > 0:
                out[names[0]] = int(ids[0])
        except Exception:
            pass
    return out


def _make_command_target(
    robot,
    joint_map: Dict[str, int],
    cmd: Dict[str, float],
    current_target: torch.Tensor,
    hand_input_mode: str,
    thumb1_from_thumb2: bool = True,
    thumb1_coupling_scale: float = 1.0,
) -> torch.Tensor:
    """Update full robot target tensor from a joint-name command dictionary.

    Arms/body: direct raw joint target, preserving the original hard-coded arm initialization.

    Hands: incoming ROS/Inspire command -> raw Isaac joint qpos target.
    This version intentionally keeps the previous master-joint-only behavior for
    the four fingers: it writes only the six bridge master channels from
    HAND_L_MAP/HAND_R_MAP and does not expand the full mimic/dependent tree.

    Thumb fix only: if thumb_1 is absent or almost zero while thumb_2 is present,
    copy thumb_2's percentage command to thumb_1 before converting to qpos. This
    lets the thumb root move without changing the behavior of index/middle/ring/little.
    """
    target = current_target.clone()

    # Work on a local copy so debug printing can show the effective command.
    cmd = {str(k): float(v) for k, v in cmd.items()}

    if thumb1_from_thumb2:
        for side in ("left", "right"):
            thumb2 = f"{side}_thumb_2_joint"
            thumb1 = f"{side}_thumb_1_joint"
            if thumb2 not in cmd:
                continue
            # Only fill thumb_1 when the incoming 6th channel is missing or zero.
            # This preserves independent thumb_1 control if the policy/bridge sends a non-zero value.
            if thumb1 not in cmd or abs(float(cmd.get(thumb1, 0.0))) <= 1e-6:
                copied = float(cmd[thumb2]) * float(thumb1_coupling_scale)
                if hand_input_mode in ("pct_close1", "pct_open1"):
                    copied = float(np.clip(copied, 0.0, 1.0))
                cmd[thumb1] = copied

    for jname, value in cmd.items():
        jid = joint_map.get(jname)
        if jid is None:
            continue

        if _is_hand_joint_name(jname):
            # Hand commands are percentages from /inspire_hand/ctrl/*, not raw radians.
            # Keep master-joint-only execution: do not expand mimic joints here.
            target[:, jid] = float(_hand_cmd_to_joint_target(jname, float(value), hand_input_mode))
        else:
            # Keep the original arm/body direct target behavior.
            target[:, jid] = float(value)

    return target

def _apply_joint_targets(robot, joint_target: torch.Tensor, debug: bool = False):
    """Apply persistent full-joint position targets to PhysX.

    In IsaacLab, set_joint_position_target() updates the articulation target buffer.
    write_data_to_sim() is required in direct-sim stepping mode; otherwise targets may not be pushed to PhysX
    before env.sim.step().
    """
    robot.set_joint_position_target(joint_target)
    try:
        robot.write_data_to_sim()
        if debug:
            print("[APPLY] set_joint_position_target + write_data_to_sim OK", flush=True)
    except Exception as e:
        if debug:
            print(f"[APPLY][WARN] write_data_to_sim failed: {e}", flush=True)




def _set_fixed_head_pitch_target(
    current_target: torch.Tensor,
    joint_map: Dict[str, int],
    head_pitch: Optional[float],
) -> Tuple[torch.Tensor, Optional[int]]:
    """Force head_pitch_joint to a fixed target so the head stays slightly lowered.

    This is intentionally applied after initialization and again every control
    loop, because incoming ROS/ZMQ commands may also contain head_pitch_joint.
    The final value should be dominated by --head_pitch.
    """
    if head_pitch is None:
        return current_target, None

    head_pitch_id = joint_map.get("head_pitch_joint")
    if head_pitch_id is None:
        return current_target, None

    current_target[:, head_pitch_id] = float(head_pitch)
    return current_target, head_pitch_id

def _debug_print_cmd_and_target(
    cmd: Dict[str, float],
    joint_map: Dict[str, int],
    robot,
    current_target: torch.Tensor,
    previous_target: Optional[torch.Tensor],
    previous_joint_pos: Optional[torch.Tensor],
    recv_index: int,
    delta_threshold: float = 1e-4,
):
    """Print a compact but useful command → target → actual joint-state trace."""
    valid_items = [(k, float(v), joint_map[k]) for k, v in cmd.items() if k in joint_map]
    invalid_items = [(k, float(v)) for k, v in cmd.items() if k not in joint_map]

    def _is_left_arm(name: str) -> bool:
        return name in (
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
        )

    def _is_right_arm(name: str) -> bool:
        return name in (
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
        )

    left_arm = [x for x in valid_items if _is_left_arm(x[0])]
    right_arm = [x for x in valid_items if _is_right_arm(x[0])]
    left_hand = [x for x in valid_items if x[0].startswith("left_")]
    right_hand = [x for x in valid_items if x[0].startswith("right_")]

    target_vec = current_target[0].detach().cpu()
    actual_vec = robot.data.joint_pos[0].detach().cpu()
    target_norm = float(torch.norm(current_target[0]).item())

    if previous_target is not None:
        target_delta = current_target[0].detach().cpu() - previous_target[0].detach().cpu()
        target_delta_norm = float(torch.norm(target_delta).item())
        changed_joint_ids = torch.nonzero(torch.abs(target_delta) > delta_threshold).flatten().tolist()
    else:
        target_delta_norm = 0.0
        changed_joint_ids = []

    if previous_joint_pos is not None:
        actual_delta = robot.data.joint_pos[0].detach().cpu() - previous_joint_pos[0].detach().cpu()
        actual_delta_norm = float(torch.norm(actual_delta).item())
        moved_joint_ids = torch.nonzero(torch.abs(actual_delta) > delta_threshold).flatten().tolist()
    else:
        actual_delta_norm = 0.0
        moved_joint_ids = []

    print(
        f"[CMD][RECV #{recv_index}] total={len(cmd)} valid={len(valid_items)} invalid={len(invalid_items)} | "
        f"L_arm={len(left_arm)} R_arm={len(right_arm)} L_hand={len(left_hand)} R_hand={len(right_hand)} | "
        f"target_norm={target_norm:.4f} target_delta={target_delta_norm:.6f} actual_delta={actual_delta_norm:.6f}",
        flush=True,
    )

    if invalid_items:
        print(f"    [WARN] invalid joint names sample={invalid_items[:8]}", flush=True)

    print("    [CMD→TARGET sample]", flush=True)
    for name, value, jid in valid_items[:16]:
        print(
            f"      id={jid:02d} {name:32s} cmd={value: .6f} target={float(target_vec[jid]): .6f} actual={float(actual_vec[jid]): .6f}",
            flush=True,
        )

    if changed_joint_ids:
        names = getattr(robot, "joint_names", [])
        sample = []
        for jid in changed_joint_ids[:12]:
            jname = names[jid] if jid < len(names) else str(jid)
            sample.append((jid, jname, float(target_delta[jid])))
        print(f"    [TARGET CHANGED] count={len(changed_joint_ids)} sample={sample}", flush=True)
    else:
        print("    [TARGET CHANGED] none above threshold", flush=True)

    if moved_joint_ids:
        names = getattr(robot, "joint_names", [])
        sample = []
        for jid in moved_joint_ids[:12]:
            jname = names[jid] if jid < len(names) else str(jid)
            sample.append((jid, jname, float(actual_delta[jid])))
        print(f"    [ACTUAL MOVED] count={len(moved_joint_ids)} sample={sample}", flush=True)
    else:
        print("    [ACTUAL MOVED] none above threshold since previous debug print", flush=True)


def _publish_robot_status(zmq_server: IsaacZmqServer, robot, joint_names: list[str], hand_input_mode: str):
    pos = _first_env_vector(robot.data.joint_pos).astype(float)
    vel = _first_env_vector(robot.data.joint_vel).astype(float) if hasattr(robot.data, "joint_vel") else np.zeros_like(pos)

    # Isaac robot joint_names should align with joint_pos order.
    if len(joint_names) != len(pos):
        joint_names = [str(i) for i in range(len(pos))]

    pos_map = {name: float(pos[i]) for i, name in enumerate(joint_names)}
    vel_map = {name: float(vel[i]) for i, name in enumerate(joint_names)}

    # Return hand state in the same 0..1 percentage convention expected by sim_deployment.py.
    finger_percentages = _finger_percentages_from_robot(robot, hand_input_mode)

    payload = {
        "joint_names": joint_names,
        "joint_pos": [pos_map[n] for n in joint_names],
        "joint_vel": [vel_map[n] for n in joint_names],
        "finger_percentages": finger_percentages,
        "time": time.time(),
    }
    zmq_server.publish_status(payload)


def _extract_rgb_from_camera_sensor(sensor) -> Optional[np.ndarray]:
    """Try several Isaac-Lab camera data layouts and return RGB uint8 HxWx3."""
    if sensor is None or not hasattr(sensor, "data"):
        return None

    data = sensor.data
    candidates = []
    try:
        if hasattr(data, "output") and isinstance(data.output, dict):
            for key in ("rgb", "rgba", "color"):
                if key in data.output:
                    candidates.append(data.output[key])
    except Exception:
        pass
    for attr in ("rgb", "rgba", "color"):
        if hasattr(data, attr):
            candidates.append(getattr(data, attr))

    for cand in candidates:
        try:
            arr = _as_np(cand)
            # Possible shapes: [N,H,W,3/4], [H,W,3/4], [N,3/4,H,W], [3/4,H,W]
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                arr = arr[..., :3]
                if arr.dtype != np.uint8:
                    # Cameras may output float in [0,1] or [0,255].
                    if float(np.nanmax(arr)) <= 1.5:
                        arr = arr * 255.0
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return np.ascontiguousarray(arr)
        except Exception:
            continue
    return None


def _find_camera_sensor(env):
    sensors = getattr(env.scene, "sensors", {})
    if not sensors:
        print("[WARN] env.scene.sensors is empty. Image publishing will use fallback black frame.")
        return None

    preferred_keywords = ["camera_head", "head", "camera", "rgb"]
    keys = list(sensors.keys())
    print(f"[INFO] Available sensors: {keys}")

    for kw in preferred_keywords:
        for key in keys:
            if kw.lower() in key.lower():
                print(f"[INFO] Using camera sensor: {key}")
                return sensors[key]

    first_key = keys[0]
    print(f"[WARN] No preferred camera sensor found. Using first sensor: {first_key}")
    return sensors[first_key]


def _get_rgb_frame(env, camera_sensor) -> np.ndarray:
    # Make sure RTX/camera buffers are refreshed when rendering is enabled.
    try:
        env.sim.render()
    except Exception:
        pass

    rgb = _extract_rgb_from_camera_sensor(camera_sensor)
    if rgb is not None:
        return rgb

    # Fallback: keep the ROS policy node from blocking forever. Replace this if your camera name is custom.
    if not hasattr(_get_rgb_frame, "warned"):
        print("[WARN] Could not extract RGB from camera sensor. Publishing black 640x480 frames as fallback.")
        _get_rgb_frame.warned = True
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_zero_env_action(env):
    """Best-effort zero action for env.step fallback. Not used by default."""
    try:
        shape = env.single_action_space.shape
        return torch.zeros((env.num_envs, *shape), device=env.device, dtype=torch.float32)
    except Exception:
        try:
            shape = env.action_space.shape
            if len(shape) == 1:
                return torch.zeros((env.num_envs, shape[0]), device=env.device, dtype=torch.float32)
            return torch.zeros(shape, device=env.device, dtype=torch.float32)
        except Exception:
            return None


def _scene_step(env):
    """Step physics without letting ActionManager overwrite direct joint targets."""
    env.sim.step()
    try:
        dt = float(env.cfg.sim.dt)
        env.scene.update(dt)
    except Exception:
        pass


def _adjust_scene_layout(env_cfg):
    """Keep your original scene adjustment: robot back, table/object down."""
    z_offset = 0.05
    x_offset = -0.17

    scene_items = list(vars(env_cfg.scene).keys())
    print(f"[INFO] Scene configuration items: {scene_items}")

    for obj_name, obj_cfg in vars(env_cfg.scene).items():
        if hasattr(obj_cfg, "init_state") and hasattr(obj_cfg.init_state, "pos"):
            old_pos = obj_cfg.init_state.pos
            if "robot" in obj_name.lower():
                obj_cfg.init_state.pos = (old_pos[0] + x_offset, old_pos[1], old_pos[2])
                print(f"[INFO] Robot ({obj_name}) moved back. New position: {obj_cfg.init_state.pos}")
            elif "terrain" in obj_name.lower() or "ground" in obj_name.lower():
                pass
            else:
                obj_cfg.init_state.pos = (old_pos[0], old_pos[1], old_pos[2] - z_offset)
                print(f"[INFO] Scene object ({obj_name}) lowered. New position: {obj_cfg.init_state.pos}")


def _initialize_robot_pose(env, robot, joint_map: Dict[str, int]):
    """Keep your original hard-coded dual-arm initialization."""
    HARDCODED_INIT_STATE = [
        -0.254162, -0.000600, 0.245700, -1.088863, -0.032118, -0.231486, -0.018677,
        -0.113035, -0.027755, -0.338698, -1.218771, -0.193617, -0.231646, 0.272586,
        0.995000, 0.994000, 0.996000, 0.993000, 0.993000, 0.000000,
        1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.000000,
    ]
    SAFE_LIFTED_STATE = [
        -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]

    target_joint_names = TIANGONG_PRO_LEFT_ARM_JOINTS + TIANGONG_PRO_RIGHT_ARM_JOINTS

    try:
        joint_ids, joint_names = robot.find_joints(target_joint_names)
    except Exception as e:
        print(f"[WARN] Cannot initialize robot pose with configured arm joints: {e}")
        return robot.data.joint_pos.clone()

    safe_state_dict = dict(zip(target_joint_names, SAFE_LIFTED_STATE))
    final_state_dict = dict(zip(target_joint_names, HARDCODED_INIT_STATE[:14]))

    safe_pos_ordered = [safe_state_dict[name] for name in joint_names]
    final_pos_ordered = [final_state_dict[name] for name in joint_names]

    safe_pos = torch.tensor(safe_pos_ordered, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
    final_pos = torch.tensor(final_pos_ordered, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
    zero_vel = torch.zeros_like(safe_pos)

    try:
        robot.data.default_joint_pos[:, joint_ids] = final_pos
    except Exception:
        pass

    # Keep ActionManager offsets aligned, in case the task still uses them internally.
    if hasattr(env, "action_manager"):
        try:
            for _, term in env.action_manager._terms.items():
                if hasattr(term, "_offset") and hasattr(term, "_joint_ids"):
                    term._offset[:] = robot.data.default_joint_pos[:, term._joint_ids]
                if hasattr(term, "_default_joint_pos") and hasattr(term, "_joint_ids"):
                    term._default_joint_pos[:] = robot.data.default_joint_pos[:, term._joint_ids]
            env.action_manager.reset()
        except Exception as e:
            print(f"[WARN] Failed to align action_manager offsets: {e}")

    try:
        robot.write_joint_state_to_sim(safe_pos, zero_vel, joint_ids=joint_ids)
        robot.set_joint_position_target(safe_pos, joint_ids=joint_ids)
        try:
            robot.write_data_to_sim()
        except Exception:
            pass
        _scene_step(env)
        print("[INFO] Robot in safe lifted pose, starting smooth transition...")

        transition_steps = 60
        for step in range(transition_steps):
            alpha = (step + 1) / transition_steps
            interpolated_pos = (1 - alpha) * safe_pos + alpha * final_pos
            robot.set_joint_position_target(interpolated_pos, joint_ids=joint_ids)
            try:
                robot.write_data_to_sim()
            except Exception:
                pass
            _scene_step(env)

        robot.data.joint_pos[:, joint_ids] = final_pos
        if hasattr(robot.data, "joint_pos_target"):
            robot.data.joint_pos_target[:, joint_ids] = final_pos
        print("[INFO] Both arms reached hard-coded initial position.")
    except Exception as e:
        print(f"[WARN] Robot initialization partially failed: {e}")

    current_target = robot.data.joint_pos.clone()

    # IMPORTANT: do not initialize hand joints from HARDCODED_INIT_STATE[14:26].
    # Those values are hand percentages for the official controller, not raw Isaac joint targets.
    # Writing them directly is what can make fingers bend outward at startup.
    # Dual-arm initialization above is unchanged.
    robot.set_joint_position_target(current_target)
    try:
        robot.write_data_to_sim()
    except Exception:
        pass
    return current_target


def main():  # noqa: C901
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    _adjust_scene_layout(env_cfg)

    desired_action_device = "tiangong_pro"
    try:
        env_cfg.use_teleop_device(desired_action_device)
    except Exception as e:
        print(f"[WARN] env_cfg.use_teleop_device failed, continuing direct-joint ZMQ control: {e}")

    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.recorders = None

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = _get_robot(env)
    robot_joint_names = list(robot.joint_names)
    print(f"[INFO] Robot has {len(robot_joint_names)} joints.")
    print(f"[INFO] First robot joints: {robot_joint_names[:20]}")

    # Build map for all bridge names that actually exist in the robot.
    joint_map = _resolve_joint_map(robot, ALL_BRIDGE_JOINT_NAMES)
    missing = [n for n in ALL_BRIDGE_JOINT_NAMES if n not in joint_map]
    print(f"[INFO] Bridge-controllable joints found: {len(joint_map)}")
    if missing:
        print(f"[WARN] Bridge joint names not found in robot, ignored: {missing}")
    print("[INFO] Arm initialization mode: ORIGINAL hard-coded dual-arm safe-lift + interpolation, unchanged.", flush=True)
    print(f"[INFO] Hand command mode: {args_cli.hand_input_mode}; pct_close1 maps cmd=1 to joint limit, pct_open1 uses original action_process formula.", flush=True)
    print(f"[INFO] Hand execution: master-joint-only for all fingers; thumb1_from_thumb2={args_cli.thumb1_from_thumb2}, scale={args_cli.thumb1_coupling_scale}", flush=True)
    hand_existing = [n for n in robot.data.joint_names if _is_hand_joint_name(n)]
    mimic_existing = [n for n in TIANGONG_PRO_MIMIC_JOINTS.keys() if n in joint_map]
    print(f"[INFO] Hand joints found in robot: {len(hand_existing)} | mimic hand joints found: {len(mimic_existing)}", flush=True)
    print(f"[INFO] Right hand joints sample: {[n for n in hand_existing if n.startswith('right_')][:24]}", flush=True)
    print(f"[INFO] Right hand mimic sample: {[n for n in mimic_existing if n.startswith('right_')][:24]}", flush=True)

    current_target = _initialize_robot_pose(env, robot, joint_map)

    # Fixed low-head posture. This keeps the head camera looking slightly down.
    # If the sign is opposite on your robot, launch with --head_pitch 0.25 instead.
    current_target, head_pitch_id = _set_fixed_head_pitch_target(current_target, joint_map, args_cli.head_pitch)
    if head_pitch_id is not None:
        robot.set_joint_position_target(current_target)
        try:
            robot.write_data_to_sim()
        except Exception:
            pass
        print(f"[INFO] Fixed head pitch enabled: head_pitch_joint id={head_pitch_id}, target={args_cli.head_pitch}", flush=True)
    else:
        print("[WARN] head_pitch_joint not found; fixed low-head posture is disabled.", flush=True)

    camera_sensor = None if args_cli.no_image else _find_camera_sensor(env)

    zmq_server = IsaacZmqServer(args_cli.cmd_port, args_cli.status_port, args_cli.image_port)
    rate_limiter = RateLimiter(args_cli.step_hz)
    rate_limiter.update_from_env(env)
    print(f"[INFO] RateLimiter sleep_duration={rate_limiter.sleep_duration:.6f}s")

    status_period = 1.0 / max(args_cli.status_hz, 1e-6)
    image_period = 1.0 / max(args_cli.image_hz, 1e-6)
    last_status_t = 0.0
    last_image_t = 0.0
    last_perf_t = time.time()
    step_count = 0
    cmd_count = 0
    total_cmd_count = 0
    last_debug_target = current_target.clone()
    last_debug_joint_pos = robot.data.joint_pos.clone()
    interrupted = False

    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] Ctrl+C detected. Cleaning up...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        while simulation_app.is_running() and not interrupted:
            loop_t0 = time.perf_counter()
            with torch.inference_mode():
                if dynamic_reset_gripper_effort_limit_sim is not None:
                    try:
                        if getattr(env.cfg, "dynamic_reset_gripper_effort_limit", False):
                            dynamic_reset_gripper_effort_limit_sim(env, desired_action_device)
                    except Exception as e:
                        if args_cli.debug_cmd:
                            print(f"[WARN] dynamic_reset_gripper_effort_limit_sim failed: {e}", flush=True)

                cmd = zmq_server.recv_latest_cmd()
                if cmd is not None:
                    cmd_count += 1
                    total_cmd_count += 1
                    current_target = _make_command_target(
                        robot,
                        joint_map,
                        cmd,
                        current_target,
                        args_cli.hand_input_mode,
                        args_cli.thumb1_from_thumb2,
                        args_cli.thumb1_coupling_scale,
                    )

                    do_debug = args_cli.debug_cmd and (
                        total_cmd_count <= 3 or total_cmd_count % max(1, args_cli.debug_cmd_every) == 0
                    )
                    if do_debug:
                        _debug_print_cmd_and_target(
                            cmd=cmd,
                            joint_map=joint_map,
                            robot=robot,
                            current_target=current_target,
                            previous_target=last_debug_target,
                            previous_joint_pos=last_debug_joint_pos,
                            recv_index=total_cmd_count,
                            delta_threshold=args_cli.debug_joint_delta,
                        )
                        last_debug_target = current_target.clone()
                        last_debug_joint_pos = robot.data.joint_pos.clone()

                # Re-apply fixed head pitch every loop so ZMQ/ROS commands cannot overwrite it.
                if head_pitch_id is not None:
                    current_target[:, head_pitch_id] = float(args_cli.head_pitch)

                apply_debug = args_cli.debug_cmd and cmd is not None and (total_cmd_count <= 3 or total_cmd_count % max(1, args_cli.debug_cmd_every) == 0)
                _apply_joint_targets(robot, current_target, debug=apply_debug)
                _scene_step(env)
                step_count += 1

                now = time.time()
                if now - last_status_t >= status_period:
                    _publish_robot_status(zmq_server, robot, robot_joint_names, args_cli.hand_input_mode)
                    last_status_t = now

                if not args_cli.no_image and now - last_image_t >= image_period:
                    rgb = _get_rgb_frame(env, camera_sensor)
                    zmq_server.publish_rgb(rgb)
                    last_image_t = now

                if now - last_perf_t >= 2.0:
                    elapsed = now - last_perf_t
                    actual_norm = float(torch.norm(robot.data.joint_pos[0]).item())
                    target_norm = float(torch.norm(current_target[0]).item())
                    track_err = float(torch.norm((current_target[0] - robot.data.joint_pos[0])).item())
                    print(
                        f"[PERF] sim_hz={step_count/elapsed:.1f} | "
                        f"cmd_recv={cmd_count/elapsed:.1f}/s | "
                        f"target_norm={target_norm:.3f} | actual_norm={actual_norm:.3f} | track_err={track_err:.3f}",
                        flush=True,
                    )
                    step_count = 0
                    cmd_count = 0
                    last_perf_t = now

            rate_limiter.sleep()

    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}\n")
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        zmq_server.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()