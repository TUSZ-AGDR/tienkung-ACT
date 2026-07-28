#  x-humanoid-training-toolchain 模仿学习训练工具链 - 用户操作手册

**版本**: v2.1.6  
**最后更新**: 2026-02-24  
**框架**: LeRobot 0.3.4+ + 1105_2 数据集  
**兼容机型**: 天工无疆(Pro)  

---

## 📖 目录

1. [项目简介](#1-项目简介)
2. [系统要求](#2-系统要求)
3. [环境配置](#3-环境配置)
4. [数据准备与转换](#4-数据准备与转换)
5. [模型训练](#5-模型训练)
6. [模型评估](#6-模型评估)
7. [真机部署](#7-真机部署)
8. [参考资源](#8-参考资源)

---

## 1. 项目简介

### 1.1 项目概述

**x-humanoid-training-toolchain 模仿学习训练工具链** 是 RoboMIND 数据集和天工行者无疆机器人对于 Lerobot 开源框架适配的训练工具链。

**核心特性**
- ✅ **多任务学习**：107k 演示跨越 479 个操作任务
- ✅ **完整数据管道**：HDF5 → LeRobotDataset（优化格式）
- ✅ **ACT 策略模型**：基于 Transformer 的动作分块及 VAE
- ✅ **ROS2 部署**：真实硬件上的实时推理
- ✅ **完整工作流支持**：包含数据转换、训练、评估和ROS2部署

**应用场景**
- 从人类演示中学习操作技能
- 快速机器人策略部署
- 多任务技能迁移和泛化
- 模仿学习研究

## 2. 系统要求

### 2.1 硬件要求

| 配置类型     | 最低要求                               | 推荐配置                      |
| ------------ | -------------------------------------- | ----------------------------- |
| **操作系统** | Ubuntu 22.04 LTS                       | Ubuntu 22.04 LTS              |
| **系统架构** | x86_64                                 | x86_64                        |
| **GPU**      | NVIDIA RTX 30系列及以上,显存16G及以上  | NVIDIA RTX 3090 / 4090 (24GB) |
| **显存**     | 16GB                                   | 24GB 或更高                   |
| **内存**     | 16GB                                   | 32GB 或更高                   |
| **磁盘空间** | 512GB                                  | 1TB SSD                       |
| **CPU**      | Intel i5 / AMD Ryzen 5                 | Intel i7 / AMD Ryzen 7 以上   |

**适配机型**：天工无疆（Pro）

**配置说明**：
- 以上最低配置基于官方参考：[x-humanoid-training-toolchain](https://github.com/Open-X-Humanoid/x-humanoid-training-toolchain)
- GPU要求：NVIDIA RTX 30系列及以上，显存16GB及以上
- 存储空间：最低512GB用于系统和基础数据，实际训练中建议1TB+以存储数据集（~16GB+）、多个训练检查点和日志
- 网络：如果git等资源拉取过慢，建议使用代理

### 2.2 软件依赖

| 组件 | 版本 | 用途 |
|:---|:---:|:---|
| **LeRobot** | 0.3.4+ | 核心训练框架 |
| **PyTorch** | 2.0+ | 深度学习引擎 |
| **Python** | 3.10+ | 编程语言 |
| **NVIDIA 驱动** | ≥ 535.x | GPU 支持 |
| **CUDA** | 12.1+ | GPU 计算（GPU驱动配套版本） |
| **FFmpeg** | ≥ 4.4 | 视频处理 |
| **HuggingFace** | 最新版 | 模型库和数据集 |

---

## 3. 环境配置

环境配置分为两个部分：**基础软件安装**（CUDA、Miniconda）和 **VLA 训练环境配置**（Python 环境、依赖库）。

### 3.1 基础软件安装

#### 3.1.1 安装 NVIDIA CUDA

CUDA 是 GPU 计算的基础，必须首先安装。

**第一步：检查 NVIDIA 驱动版本**

```bash
# 确保已安装最新 NVIDIA 驱动
nvidia-smi

# 输出中会显示驱动版本和配套的 CUDA 版本
# 例如：CUDA Version: 12.8
```

**第二步：下载并安装 CUDA Toolkit**

根据 nvidia-smi 显示的 CUDA 版本，从官方下载对应的 CUDA Toolkit。以 CUDA 12.8 为例：

```bash
# 创建软件安装目录
mkdir -p ~/GTM/soft
cd ~/GTM/soft

# 从 NVIDIA 官方下载 CUDA Toolkit
# 访问：https://developer.nvidia.com/cuda-toolkit-archive 选择对应版本
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run

# 执行安装
sudo sh cuda_12.8.0_570.86.10_linux.run
```

**第三步：设置 CUDA 环境变量**

```bash
# 编辑 ~/.bashrc 文件，添加 CUDA 路径
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# 立即生效
source ~/.bashrc

# 验证安装成功
nvcc --version  # 应显示 CUDA Compiler 版本
```

#### 3.1.2 安装 Miniconda

Miniconda 是轻量级的包管理器，用于创建隔离的 Python 环境。

```bash
# 下载 Miniconda 安装脚本
cd ~/GTM/soft
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 执行安装（全程一直选择 yes）
bash ./Miniconda3-latest-Linux-x86_64.sh

# 安装完成后验证 .bashrc 中是否包含 conda 初始化配置
cat ~/.bashrc

# 重新加载 shell 配置使 conda 生效
source ~/.bashrc

# 验证 conda 安装成功
conda --version
```

#### 3.1.3 安装系统依赖

```bash
# 安装 FFmpeg（视频处理）
sudo apt install ffmpeg

# 在 conda 中安装 h5py（HDF5 数据格式支持，用于读取数据集）
conda install -c conda-forge h5py -y
```

---

### 3.2 VLA 训练环境配置

#### 3.2.1 解压代码并进入项目目录

```bash
# 解压 VLA 项目代码，x-humanoid-training-toolchain_20260224v2.1.6.zip根据具体情况版本号会有变化
cd ~/GTM/soft
unzip -d .temp_dir x-humanoid-training-toolchain_20260224v2.1.6.zip && mv .temp_dir/*/* . && rm -rf .temp_dir 

# 进入项目目录
cd ~/GTM/soft/x-humanoid-training-toolchain
```

#### 3.2.2 创建 VLA 环境

```bash
# 创建名为 env_vla 的 Python 3.10 环境
conda create -n env_vla python=3.10 -y

# 激活环境
conda activate env_vla

# 验证 Python 版本
python --version  # 应显示 3.10.x
```

**⚠️ 重要提示：后续所有命令都需要在 `env_vla` 环境中执行。如果新开 terminal，需要重新运行 `conda activate env_vla`**

#### 3.2.3 安装 PyTorch

PyTorch 版本必须与 CUDA 版本匹配。根据你的 CUDA 版本选择相应的安装命令。

**CUDA 12.8 版本（推荐）：**
```bash
pip install -U --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128/ \
    torch torchvision torchaudio
```

**CUDA 12.1 版本：**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**验证 PyTorch 安装：**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
# 预期输出：CUDA Available: True
```

#### 3.2.4 安装 VLA 工具链及依赖

在项目根目录中安装 VLA 框架：

```bash
# 以开发模式安装 VLA（会自动安装 LeRobot 及其依赖）
pip install -e .

# 安装额外的视频编解码依赖（版本需与 CUDA 匹配）
pip install -U --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128/ torchcodec

# 验证 VLA 安装成功
python -c "from deployment.action_policy import PolicyAgent; print('✓ VLA 环境配置完成')"
```

#### 3.2.5 下载 1105_2 数据集（可选）

如果需要进行训练，下载 1105_2 数据集（约 17GB）：

```bash
# 创建数据存储目录
mkdir -p ~/GTM/dataset/

# 注意：1105_2 数据集为私有数据集，需要通过授权渠道获取，获取后放入~/GTM/dataset目录下
# 数据集获取方式请联系项目负责人或查看项目文档

# 验证下载完成
ls ~/GTM/dataset/1105_2/
# 应该包含：1105_112157/ ,..., 1105_122009/ 等
```

---

## 4. 数据准备与转换

### 4.1 理解数据格式

**1105_2 数据集结构：**
- 格式：HDF5 文件
- 总大小：~16GB+

### 4.2 运行数据转换

项目已提供了自动化转换脚本 `convert.sh`，可以快速将 HDF5 数据转换为 LeRobot 格式。

**第一步：进入脚本目录**

```bash
cd ~/GTM/soft/x-humanoid-training-toolchain/scripts/
```

**第二步：赋予脚本执行权限**

```bash
chmod +x convert.sh
```

**第三步：运行转换脚本**

```bash
./convert.sh
```

### 4.3 验证转换结果

```bash
# 检查输出结构
ls -la ~/GTM/data/dataset/pick_place/

# 预期输出：
# data/
# meta/
# videos/
# images/(转换临时文件夹)
```

---

## 5. 模型训练

### 5.1 编辑训练配置

**✅ 推荐方式：自动替换用户名**

项目提供的 `scripts/configs/train_config.json` 中包含 `[username]` 占位符，需要替换为实际的用户名。可以使用 `sed` 命令自动完成：

```bash
# 进入项目目录
cd ~/GTM/soft/x-humanoid-training-toolchain

# 使用 sed 自动替换 [username] 为当前用户名
sed -i "s/\[username\]/$USER/g" scripts/configs/train_config.json

# 验证替换成功
cat scripts/configs/train_config.json | grep "root"
# 应该看到："/home/yourusername/GTM/data/dataset/pick_place"
```

### 5.2 训练

**第一步：激活环境和配置**

```bash
# 激活 VLA 环境
conda activate env_vla

# 进入项目目录
cd ~/GTM/soft/x-humanoid-training-toolchain

# 自动替换配置中的用户名
sed -i "s/\[username\]/$USER/g" scripts/configs/train_config.json
```

**第二步：启动训练**

```bash
# ✅ 推荐：使用配置文件启动训练
python lerobot/scripts/train.py --config_path=scripts/configs/train_config.json
```

**预期结果**：
- 训练进程开始，显示进度条和训练指标
- 检查点将保存到配置的输出目录，例如~/GTM/data/output/pick_place/checkpoints/040000

---

## 6. 模型评估

### 6.1 评估训练好的模型

**基本评估命令：**

```bash
# 使用LeRobot评估脚本
# 在环境上评估模型，需要安装gym等环境包，见factory.py L28-41
python lerobot/scripts/eval.py \
    --policy.path=~/GTM/data/output/pick_place/checkpoints/last/pretrained_model \
    --env.type=你的环境类型[pusht/aloha/xarm]

# 参考示例代码
python examples/2_evaluate_pretrained_policy.py
```

```bash
# 使用model_eval.py评估脚本，注意修改其中pretrained_model_path及dataset_path
python eval/model_eval.py
```

---

## 7. 真机部署

本章节介绍如何将训练好的模仿学习模型部署到天工无疆机器人真机上。部署过程分为两个阶段：**算力板环境准备**和**模型运行**。

### 7.1 网络配置

#### 7.1.1 连接机器人

1. **使用网线连接机器人**
   - 将网线从计算机连接到机器人的以太网接口

2. **配置本地网络**
   ```bash
   # 本机 IP 配置
   IP: 192.168.41.10
   子网掩码: 255.255.255.0
   ```

3. **登录算力板**
   ```bash
   # SSH 登录到机器人算力板
   ssh nvidia@192.168.41.2
   # 密码：nvidia
   ```

4. **连接 Wi-Fi（可选，用于访问外网）**
   ```bash
   # 查看可用的 Wi-Fi 热点
   sudo nmcli device wifi list
   
   # 连接指定的 Wi-Fi
   sudo nmcli device wifi connect 'Wi-Fi SSID' password 'Wi-Fi 密码'
   
   # 检查是否连接成功
   ifconfig
   ```

---

### 7.2 准备软件包

在算力板上创建目录结构并上传所需文件：

```bash
# 在算力板上创建目录
mkdir -p ~/GTM/checkpoints
mkdir -p ~/GTM/soft

# 目录结构如下：
# GTM
#  ├── 📂 checkpoints/                                          # 模型目录
#  │   ├── 040000/                                              # 训练好的模型（示例）
#  │       └── pretrained_model/
#  ├── 📂 soft/                                                 # 软件包
#  │   ├── 📜 x-humanoid-training-toolchain_20260224v2.1.6.zip  # VLA 代码包

```

**需要准备的文件：**
1. `x-humanoid-training-toolchain_20260224v2.1.6.zip` - VLA 代码包，# 解压 VLA 项目代码，代码包版本号根据具体情况会有变化
2. 训练好的模型文件（从训练阶段的输出目录复制），例如训练机器上的~/GTM/data/output/pick_place/checkpoints/040000/pretrained_model

---

### 7.3 算力板环境配置

#### 7.3.1 安装 Miniconda（ARM64 版本）

```bash
# 进入软件包目录
cd ~/GTM/soft

# 下载 ARM64 版本的 Miniconda
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_25.11.1-1-Linux-aarch64.sh

# 执行安装
bash ./Miniconda3-py310_25.11.1-1-Linux-aarch64.sh

# 按照提示完成安装，最后选择 yes 初始化 conda

# 重新加载 shell 配置
source ~/.bashrc

# 验证 conda 安装
conda --version
```

#### 7.3.2 解压代码

```bash
# 解压 VLA 工具链代码
cd ~/GTM/soft/
unzip -d .temp_dir x-humanoid-training-toolchain_20260224v2.1.6.zip && mv .temp_dir/*/* . && rm -rf .temp_dir 

```

#### 7.3.3 创建 VLA 部署环境

```bash
# 进入项目目录
cd ~/GTM/soft/x-humanoid-training-toolchain

# 安装系统依赖
sudo apt install ffmpeg -y

# 创建 Python 3.10 环境
conda create -n env_vla python=3.10 -y

# 激活环境
conda activate env_vla

# 安装 VLA 工具链
pip install -e .

# 安装 PyTorch（Jetson 专用）
pip install ~/GTM/soft/Packages/torch-2.8.0-cp310-cp310-linux_aarch64.whl
pip install ~/GTM/soft/Packages/torchcodec-0.6.0-cp310-cp310-linux_aarch64.whl
pip install ~/GTM/soft/Packages/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl
# 安装 ROS2 消息包
sudo dpkg -i ~/GTM/soft/Packages/ros-humble-bodyctrl-msgs_0.0.0-0jammy_arm64.deb
```

---

### 7.4 配置部署脚本

#### 7.4.1 修改模型路径

编辑部署脚本中的模型路径：

```bash
# 编辑部署脚本
nano ~/GTM/soft/x-humanoid-training-toolchain/deployment/ros2_deployment_inspire.py
```

**找到第 42 行，修改为你的模型路径：**
```python
# 原始代码（第 42 行）
model_path = "PATH_TO_MODEL"

# 修改为（使用绝对路径或相对路径）
model_path = "~/GTM/checkpoints/040000/pretrained_model"
```

#### 7.4.2 配置相机话题

根据机器人版本和相机配置，修改相机话题名称：

```bash
# 检查当前的相机话题
ros2 topic list | grep "camera"
```

**编辑部署脚本（第 91-96 行）：**

```python
# 根据相机话题信息确定camera_name ，注意：这个值可能会跟随SDK版本有变化，因此以真实机器人相机话题为准！
# SDK版本 2.0.6 之前，一般都是
camera_name = 'camera'   
# SDK版本 2.0.6 及之后，一般都是
camera_name = 'ob_camera_head'

# 订阅相机话题
self.rgb_sub = Subscriber(self, Image, f'/{camera_name}/color/image_raw')
self.depth_sub = Subscriber(self, Image, f'/{camera_name}/depth/image_raw')
```

**保存修改：**
- 按 `Ctrl + O` 保存
- 按 `Ctrl + X` 退出

**保存修改后退出。**

---

### 7.5 准备机器人执行环境

在运行模仿学习程序之前，需要手动操作机器人进入正确的工作状态。

#### 7.5.1 机器人初始化流程

**步骤 1：开启机器人**
- 打开机器人电源

**步骤 2：遥控器复位**
- 将遥控器**所有拨杆**拨回中间位置

**步骤 3：机器人回零**
- 按遥控器 **D 键**进行回零操作
- 等待机器人落地后**保持稳定 20 秒**

**步骤 4：机器人站立**
- 长按遥控器 **A 键**，使机器人站立

**步骤 5：启用半身运动控制模式**
- **F 键下拨** → **长按 A 键** → **E 键上拨**
- 成功后机器人上半身可自由控制

**步骤 6：调整头部角度**

使用 ROS2 命令调整头部姿态：
```bash
ros2 topic pub /head/cmd_pos bodyctrl_msgs/msg/CmdSetMotorPosition \
  "{cmds: [
    {name: 1, pos: 0.0, spd: 0.2, cur: 5.0},
    {name: 2, pos: 0.4, spd: 0.2, cur: 5.0},
    {name: 3, pos: 0.0, spd: 0.2, cur: 5.0}
  ]}"
```

#### 7.5.2 启动相机

根据机器人版本选择对应的相机启动方式：

**SDK版本 2.0.6 之前：**
```bash
# 进入相机驱动目录
cd /home/nvidia/orbbec_camera_ros2

# 加载 ROS2 环境
source install/setup.bash

# 启动相机节点
ros2 launch orbbec_camera gemini_330_series.launch.py

# 验证相机启动成功
ros2 topic list | grep camera
```

**SDK版本 2.0.6 及之后：**
```bash
# 进入相机驱动目录
cd /home/nvidia/orbbec_camera_ros2

# 加载 ROS2 环境
source install/setup.bash

# 使用 systemd 服务启动相机
sudo systemctl start orbbec_head.service

# 验证相机启动成功
ros2 topic list | grep camera
```

---

### 7.6 运行模仿学习程序

#### 7.6.1 首次测试运行（无障碍物）

**目的：验证机器人手臂能够正常抬起和执行动作**

```bash
# 步骤 1：移开前方桌子和障碍物（确保机器人活动空间无阻碍）

# 步骤 2：激活 VLA 环境
conda activate env_vla

# 步骤 3：进入项目目录
cd ~/GTM/soft/x-humanoid-training-toolchain

# 步骤 4：运行部署脚本
python deployment/ros2_deployment_inspire.py

# 步骤 5：观察机器人动作
# - 机器人应该抬起手臂
# - 如果动作正常，按 Ctrl+C 停止程序
```

**预期结果：**
- 机器人手臂抬起并执行基本动作
- 无异常报错或卡顿

#### 7.6.2 正式执行（完整场景）

```bash
# 步骤 1：搬回桌子并布置场景
# - 根据训练数据中的场景布置物品
# - 确保物品位置与训练场景相似

# 步骤 2：再次运行模仿学习程序
conda activate env_vla
cd ~/GTM/soft/x-humanoid-training-toolchain
python deployment/ros2_deployment_inspire.py

# 步骤 3：观察机器人执行任务
# - 机器人将根据视觉输入执行学习到的动作
# - 实时观察动作是否符合预期

# 步骤 4：停止程序
# 按 Ctrl+C 安全停止
```

---

### 7.7 部署注意事项

#### 7.7.1 安全注意事项

⚠️ **重要安全提示：**
1. **保持安全距离**：在机器人运动时，人员应保持安全距离（建议 2 米以上）
2. **紧急停止**：熟悉遥控器的紧急停止按钮位置
3. **场景一致性**：确保真机场景与训练场景尽可能相似
4. **监控运行**：首次运行时应全程监控，观察是否有异常动作
5. **电量充足**：确保机器人电量充足，避免运行中断

#### 7.7.2 常见问题排查

**问题 1：找不到模型文件**
```bash
# 检查模型路径是否正确
ls -la ~/GTM/checkpoints/040000/pretrained_model

# 确认 ros2_deployment_inspire.py 中的路径配置
cat deployment/ros2_deployment_inspire.py | grep "model_path"
```

**问题 2：相机话题无数据**
```bash
# 检查相机是否启动
ros2 topic list | grep camera

# 测试相机数据流
ros2 topic echo /camera/color/image_raw --once
# 或
ros2 topic echo /ob_camera_head/color/image_raw --once
```

**问题 3：ROS2 消息包缺失**
```bash
# 重新安装 ROS2 消息包
sudo dpkg -i ~/GTM/soft/Packages/ros-humble-bodyctrl-msgs_0.0.0-0jammy_arm64.deb

# 验证安装
ros2 interface show bodyctrl_msgs/msg/CmdSetMotorPosition
```

**问题 4：PyTorch CUDA 不可用**
```bash
# 在 env_vla 环境中验证 PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# 如果输出 False，检查 Jetson CUDA 环境
nvcc --version
```

---

### 7.8 部署流程总结

**完整部署检查清单：**

- [ ] 网络连接配置完成（网线 + Wi-Fi）
- [ ] 软件包上传到 `~/GTM/` 目录
- [ ] Miniconda（ARM64）安装成功
- [ ] `env_vla` 环境创建并安装依赖
- [ ] ROS2 消息包安装完成
- [ ] 模型文件复制到算力板
- [ ] 部署脚本中的模型路径配置正确
- [ ] 相机话题名称配置正确
- [ ] 机器人已进入半身运动控制模式
- [ ] 相机节点启动成功
- [ ] 首次无障碍测试通过
- [ ] 场景布置完成
- [ ] 安全措施到位

**快速启动命令（部署完成后）：**
```bash
# 1. SSH 登录算力板
ssh nvidia@192.168.41.2

# 2. 激活环境
conda activate env_vla

# 3. 进入项目目录
cd ~/GTM/soft/x-humanoid-training-toolchain

# 4. 运行部署脚本
python deployment/ros2_deployment_inspire.py
```

---

## 8. 参考资源

### 官方文档
- **LeRobot 官方仓库**: https://github.com/huggingface/lerobot
- **X-Humanoid 训练工具链**: https://github.com/Open-X-Humanoid/x-humanoid-training-toolchain

### 相关论文
- **ACT (Action Chunking with Transformers)**: https://arxiv.org/abs/2304.13705

---

## 常用命令速查表

```bash
# 1. 激活环境
conda activate env_vla

# 2. 数据转换
cd ~/GTM/soft/x-humanoid-training-toolchain/scripts/
chmod +x convert.sh
./convert.sh

# 3. 准备训练配置
cd ~/GTM/soft/x-humanoid-training-toolchain
sed -i "s/\[username\]/$USER/g" scripts/configs/train_config.json

# 4. 启动训练
python lerobot/scripts/train.py --config_path=scripts/configs/train_config.json

# 5. 真机部署（算力板）
# (配置好 deployment/ros2_deployment_inspire.py 中的模型路径后)
cd ~/GTM/soft/x-humanoid-training-toolchain
python deployment/ros2_deployment_inspire.py
```

---
<div align="center">
  <b>🚀 开始您的人形机器人模仿学习之旅!</b>
  <br><br>
  Made with ❤️ by ECBG Humanoid Team
</div>
