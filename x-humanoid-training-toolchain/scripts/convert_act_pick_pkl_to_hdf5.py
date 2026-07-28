#!/usr/bin/env python
# -*- coding: utf-8 -*-
# [zengquan] 将 act_pick 采集的 pkl 格式数据转换为项目标准的 trajectory.hdf5 格式

import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def parse_timestamp(filename: str) -> float:
    """[zengquan] 从 pkl 文件名解析时间戳，文件名格式：2026-06-27T16_54_15.411537.pkl"""
    ts_str = filename.replace(".pkl", "").replace("_", ":")
    dt = datetime.fromisoformat(ts_str)
    return dt.timestamp()


def create_signal_group(parent: h5py.Group, name: str, data: np.ndarray, timestamps: np.ndarray):
    """[zengquan] 创建符合项目规范的信号组，包含 data/timestamp/is_intervene 三个数据集"""
    group = parent.create_group(name)
    group.create_dataset("data", data=data)
    group.create_dataset("timestamp", data=timestamps)
    group.create_dataset("is_intervene", data=np.zeros(len(data), dtype=bool))


def convert_episode(ep_dir: Path, out_path: Path) -> bool:
    """[zengquan] 将单个轨迹目录下的所有 pkl 文件合并转换为一个 trajectory.hdf5"""
    pkl_files = sorted([f for f in ep_dir.iterdir() if f.suffix == ".pkl"])
    if not pkl_files:
        return False

    # [zengquan] 加载所有帧并记录时间戳
    frames = []
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            frame = pickle.load(f)
        timestamp = parse_timestamp(pkl_file.name)
        frames.append((timestamp, frame))

    # [zengquan] 按时间戳排序，确保轨迹时序正确
    frames.sort(key=lambda x: x[0])
    timestamps = np.array([t for t, _ in frames], dtype=np.float64)
    n_frames = len(frames)

    # [zengquan] 提取 puppet 机器人关节数据
    arm_left = np.stack([f["arm_joints"]["left"] for _, f in frames]).astype(np.float32)
    arm_right = np.stack([f["arm_joints"]["right"] for _, f in frames]).astype(np.float32)
    hand_left = np.stack([f["hand_joints"]["left"] for _, f in frames]).astype(np.float32)
    hand_right = np.stack([f["hand_joints"]["right"] for _, f in frames]).astype(np.float32)

    # [zengquan] 提取力传感器数据
    force_left = np.array([f["force_sensor"]["left"] for _, f in frames], dtype=np.float32)
    force_right = np.array([f["force_sensor"]["right"] for _, f in frames], dtype=np.float32)

    # [zengquan] 提取 master 控制信号
    control_arm_left = np.stack([f["control"]["arm"]["position"]["left"] for _, f in frames]).astype(np.float32)
    control_arm_right = np.stack([f["control"]["arm"]["position"]["right"] for _, f in frames]).astype(np.float32)
    control_hand_left = np.stack([f["control"]["hand"]["position"]["left"] for _, f in frames]).astype(np.float32)
    control_hand_right = np.stack([f["control"]["hand"]["position"]["right"] for _, f in frames]).astype(np.float32)

    # [zengquan] 提取图像数据，保持原始压缩字节格式存储
    color_images = [np.array(f["images"]["head"], dtype=np.uint8) for _, f in frames]
    depth_images = [np.array(f["depths"]["head"], dtype=np.uint8) for _, f in frames]

    # [zengquan] 创建输出目录并写入 hdf5
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as hf:
        # [zengquan] puppet 组：实际机器人执行数据
        puppet = hf.create_group("puppet")
        create_signal_group(puppet, "arm_left_position_align", arm_left, timestamps)
        create_signal_group(puppet, "arm_right_position_align", arm_right, timestamps)
        create_signal_group(puppet, "end_effector_left_position_align", hand_left, timestamps)
        create_signal_group(puppet, "end_effector_right_position_align", hand_right, timestamps)

        # [zengquan] master 组：控制端输入数据
        master = hf.create_group("master")
        create_signal_group(master, "arm_left_position_align", control_arm_left, timestamps)
        create_signal_group(master, "arm_right_position_align", control_arm_right, timestamps)
        create_signal_group(master, "end_effector_left_position_align", control_hand_left, timestamps)
        create_signal_group(master, "end_effector_right_position_align", control_hand_right, timestamps)

        # [zengquan] sensor 组：额外传感器数据
        sensor = hf.create_group("sensor")
        create_signal_group(sensor, "force_left", force_left, timestamps)
        create_signal_group(sensor, "force_right", force_right, timestamps)

        # [zengquan] camera_observations 组：相机图像数据
        camera = hf.create_group("camera_observations")
        color_group = camera.create_group("color_images")
        # [zengquan] 使用 uint8 变长数组存储压缩图像字节，避免 VLEN string 不支持 NULL 字节的问题
        color_group.create_dataset("camera_head", data=color_images, dtype=h5py.vlen_dtype(np.uint8))
        depth_group = camera.create_group("depth_images")
        depth_group.create_dataset("camera_head", data=depth_images, dtype=h5py.vlen_dtype(np.uint8))
        camera.create_dataset("timestamp", data=timestamps)
        camera.create_dataset("is_intervene", data=np.zeros(n_frames, dtype=bool))

        # [zengquan] metadata 组：轨迹元数据
        meta = hf.create_group("metadata")
        meta.create_dataset("collection_time", data=datetime.now().isoformat())
        meta.create_dataset("collector", data="act_pick_converter")
        meta.create_dataset("data_format_version", data="1.0.0")
        meta.create_dataset("data_type", data="pkl_to_hdf5")
        meta.create_dataset("language_instruction", data="")
        meta.create_dataset("trajectory_length", data=n_frames)

    return True


def main():
    """[zengquan] 命令行入口：批量转换 act_pick 数据"""
    parser = argparse.ArgumentParser(
        description="[zengquan] Convert act_pick pkl episodes to project standard trajectory.hdf5"
    )
    parser.add_argument("--src_root", type=str, required=True, help="act_pick 数据根目录")
    parser.add_argument("--out_root", type=str, required=True, help="输出 hdf5 根目录")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    ep_dirs = [d for d in src_root.iterdir() if d.is_dir()]
    print(f"[zengquan] Found {len(ep_dirs)} episodes to convert")

    for ep_dir in tqdm(ep_dirs, desc="[zengquan] Converting episodes"):
        out_path = out_root / ep_dir.name / "trajectory.hdf5"
        try:
            if convert_episode(ep_dir, out_path):
                print(f"[zengquan] Converted: {ep_dir.name} -> {out_path}")
        except Exception as e:
            print(f"[zengquan] Failed to convert {ep_dir.name}: {e}")


if __name__ == "__main__":
    main()
