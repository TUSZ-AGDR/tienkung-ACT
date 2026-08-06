# TienKung ACT — 天工人形机器人具身智能数据采集与ACT训练部署工具链

## 📌 项目简介

本项目基于 **ACT（Action Chunking with Transformers）** 模仿学习算法与 **LeRobot** 框架，为天工人形机器人提供一套 **"数据采集 → 模型训练 → 仿真部署 → 真机部署"** 的具身智能工具链。

通过同构臂遥操作采集真实世界的专家演示数据，转换为 LeRobot 格式，训练 ACT 模型，最终部署到 Isaac Sim 仿真环境或天工机器人真机。

## 🏗️ 项目架构与目录结构

```
tienkung_act/
├── x-humanoid-training-toolchain/        # 核心训练与部署工具链
│   ├── lerobot/                          # LeRobot 框架
│   ├── scripts/                          # 数据转换脚本
│   ├── deployment/                       # 模型部署
│   ├── examples/                         # 使用示例
│   ├── eval/model_eval.py                # 基于数据集 Loss 的模型评估
│   ├── docker/                           # Dockerfile（cpu / gpu / gpu-dev）
│   ├── tests/                            # 单元测试
│   └── pyproject.toml                    # Python 包定义
├── TienKung_pro_sim/                     # Isaac Sim 仿真包
│   └── TienKung_pro_sim_v0.1/
│       ├── leisaac_v0.1/                 # 仿真环境（LeIsaac）
│       └── ros-humble-bodyctrl-msgs_*.deb # ROS2 自定义消息包
├── Dockerfile.save                       # 训练/部署 Docker 镜像定义
├── requirements-docker.txt               # 容器 Python 依赖
├── fastrtps_local.xml                    # FastDDS 本机通信配置
└── docs/                                 # 用户手册 / 开发者指南
```

## 🛠️ 环境搭建

### 宿主机环境（训练机）

**系统要求**：Ubuntu 22.04 、NVIDIA RTX 30 系及以上（16GB 显存起步，推荐 24GB）。

**1. 安装 NVIDIA 驱动**

**2. 安装 NVIDIA Container Toolkit**

**3. 安装 Docker 与 Miniconda**

**4. 宿主机 Isaac Sim / Isaac Lab 配置（仿真用）**

# 启动容器
docker run -it --gpus all \
  --name tienkung_act \
  --network host \
  --ipc=host \
  -v /home/ubuntu/x-humanoid-training-toolchain_20260224v2.1.6:/workspace \
  -e ROS_DOMAIN_ID=146 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/fastrtps_local.xml \
  test \
  bash

# 后续重新进入容器
docker exec -it tienkung_act bash

> **Note**：容器内已内置 ROS 2 Humble 环境，将 FastDDS 限定在 `127.0.0.1` 本机通信，避免多机干扰。

**算力板（真机部署，ARM64）环境**：在机器人算力板上安装 ARM64 版 Miniconda，并通过 Jetson 专用 PyTorch wheel 安装依赖，详见 [真机部署](#-真机部署-real-robot)。

---

## 🔄 数据采集与格式转换

### 数据采集（遥操作）

使用同构臂遥操作机器人采集专家演示数据，单帧数据以 `.pkl` 格式存储。

### 数据转换（pkl → HDF5 → LeRobot）

**转换后数据集结构**：

```
lerobot_data/pick_place/
├── data/          # parquet 帧数据
├── meta/          # 元数据（episodes、stats、tasks）
├── videos/        # 压缩视频
└── images/        # 图像（转换临时目录）
```

---

## 🏋️ 模型训练

### 启动训练

```bash
docker exec -it tienkung_act bash
cd /workspace/x-humanoid-training-toolchain

# 启动训练
python lerobot/scripts/train.py --config_path=scripts/configs/train_config.json
```
---

## 🤖 模型部署

### 仿真部署 
仿真闭环部署架构：**Isaac Sim ↔ ZMQ 桥接 ↔ ROS 2 ↔ ACT 推理**。

**三终端启动流程**：

```bash
# ── 终端 A：启动 Isaac Sim 仿真 ──
cd TienKung_pro_sim/TienKung_pro_sim_v0.1/leisaac_v0.1
./start_sim.sh
```

```bash
# ── 终端 B：Docker 内启动 ZMQ ↔ ROS2 桥接 ──
docker exec -it tienkung_act bash
cd /workspace/TienKung_pro_sim/TienKung_pro_sim_v0.1/leisaac_v0.1
./ros2_bridge.sh
```

```bash
# ── 终端 C：Docker 内启动 ACT 策略推理 ──
docker exec -it tienkung_act bash
cd /workspace/x-humanoid-training-toolchain
python deployment/sim_deployment.py \
  --model-path=/workspace/x-humanoid-training-toolchain/data/output/checkpoints/200000/pretrained_model
```

---

### 真机部署 

**1. 网络与算力板配置**

**2. 配置部署脚本**

**3. 机器人初始化流程**

1. 开启机器人电源
2. 遥控器所有拨杆拨回**中间位置**（复位）
3. 按遥控器 **D 键** 回零，落地后保持稳定 **20 秒**
4. 长按 **A 键** 使机器人站立
5. 启用半身运动控制：**F 键下拨 → 长按 A 键 → E 键上拨**
6. 调整头部角度：

```bash
ros2 topic pub /head/cmd_pos bodyctrl_msgs/msg/CmdSetMotorPosition \
  "{cmds: [
     {name: 1, pos: 0.0, spd: 0.2, cur: 5.0},
     {name: 2, pos: 0.4, spd: 0.2, cur: 5.0},
     {name: 3, pos: 0.0, spd: 0.2, cur: 5.0}
   ]}"
```

**4. 启动相机**

**5. 一键推理**

```bash
conda activate env_vla
cd ~/GTM/soft/x-humanoid-training-toolchain
python deployment/ros2_deployment_inspire.py
```

> ⚠️ **安全提示**：首次运行请移开机器人前方障碍物做无障碍测试；正式执行时人员与机器人保持 2 米以上安全距离，熟悉急停按钮，并确保场景布置与训练数据尽可能一致。

## 📚 文档资源

| 文档 | 说明 |
|------|------|
| [`docs/UserGuide.md`](docs/UserGuide.md) | 完整用户操作手册（环境配置 / 转换 / 训练 / 部署） |
| [`docs/DeveloperGuide.md`](docs/DeveloperGuide.md) | 开发者指南（架构 / 数据管道 / 策略实现） |
| [`x-humanoid-training-toolchain/deployment/readme.md`](x-humanoid-training-toolchain/deployment/readme.md) | 26 维动作向量与 `publish_action` 映射规范 |
| `TienKung_pro_sim/.../vla_sim.md` | 仿真虚拟环境安装与调试说明 |

## 🙏 致谢

- [LeRobot](https://github.com/huggingface/lerobot) — HuggingFace 机器人学习框架
- [ACT](https://arxiv.org/abs/2304.13705) — Action Chunking with Transformers
- [RoboMIND](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND) — 多本体操作数据集
- Isaac Sim / Isaac Lab — NVIDIA 仿真平台
- ROS 2 Humble — 机器人操作系统

## 📄 许可证

本项目基于 [Apache 2.0 License](x-humanoid-training-toolchain/LICENSE) 开源。
