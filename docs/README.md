# x-humanoid-training-toolchain 模仿学习训练工具链 - README

<div align="center">
  <img src="media/tutorial/vla_overview.png" width="600" alt="VLA 项目概览">
  <br>
  <b><span style="font-size:1.2em;">基于x-humanoid-training-toolchain的端到端人形机器人行为克隆框架</span></b>
  <br><br>
  
  [![LeRobot](https://img.shields.io/badge/LeRobot-0.3.4+-blue.svg)](https://huggingface.co/docs/lerobot)
  [![Dataset](https://img.shields.io/badge/Dataset-1105_2-orange.svg)]
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
  [![CUDA](https://img.shields.io/badge/CUDA-12.1%2F12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
  [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/license-Apache%202.0-yellow.svg)](LICENSE)
</div>

---

## 🚀 项目简介

**x-humanoid-training-toolchain 模仿学习训练工具链** 是 RoboMIND 数据集和天工行者无疆机器人对于 LeRobot 开源框架适配的训练工具链。

### ✨ 核心特性

- ✅ **行为克隆学习**：直接从真实世界演示中学习机器人控制策略
- ✅ **多任务学习**：支持多任务训练（任务集合由数据集定义）
- ✅ **HDF5 数据管道**：高效的数据加载和预处理流程
- ✅ **分布式训练**：支持多GPU并行训练和混合精度
- ✅ **ROS2 部署**：完整的实机部署流程和安全机制

### 🎯 应用场景

| 场景 | 说明 |
|-----|------|
| **多任务学习** | 从多个真实世界任务演示中学习通用策略 |
| **行为克隆研究** | 探索端到端学习和数据驱动方法 |
| **真机部署** | 将训练策略部署到真实 Walker-TienKung 机器人 |
| **教学科研** | 适合高校机器学习和机器人学课程 |

---

## 📊 数据和性能展示

### 数据集说明

| 指标 | 说明 |
|-----|-----|
| **数据集** | 私有 1105_2 多任务数据集（非开源，需授权；任务集合由数据提供方定义） |
| **机器人** | Walker-TienKung v2.0 |
| **行为源** | 真实世界操作员演示 |
| **数据格式** | HDF5 （图像、动作、状态） |

---

## 🛠️ 系统要求

### 推荐配置

| 项目 | 要求 |
|-----|------|
| **操作系统** | Ubuntu 22.04 (x86) |
| **GPU** | NVIDIA RTX 30系列及以上，显存 16GB+ |
| **内存** | 16GB+ |
| **磁盘** | 512GB+ |
| **Python** | 3.10+ |

**网络配置**：如果 git 等资源拉取过慢，建议使用代理
**推荐软件环境**：Ubuntu 22.04 + 与Nvidia GPU驱动版本配套的CUDA + miniConda + Python 3.10
---

## 📦 快速安装与开始

详细的安装步骤、环境配置和快速开始指南请参考相应文档：

| 阶段 | 文档 | 内容概述 |
|-----|------|---------|
| **环境配置** | [UserGuide.md](UserGuide.md#3-环境配置) | CUDA 安装、Miniconda 配置、Python 环境搭建 |
| **数据转换** | [UserGuide.md](UserGuide.md#4-数据准备与转换) | HDF5 → LeRobot 格式转换、配置文件说明 |
| **训练流程** | [UserGuide.md](UserGuide.md#5-模型训练) | 单任务/多任务训练、参数调优 |
| **评估与部署** | [UserGuide.md](UserGuide.md#6-模型评估)、[UserGuide.md](UserGuide.md#7-真机部署) | 模型评估、ROS2 部署步骤 |
| **架构深入** | [DeveloperGuide.md](DeveloperGuide.md) | 系统架构、数据管道、策略模型、性能优化 |

---

## 📚 文档资源

| 文档 | 内容 | 适用对象 |
|-----|------|---------|
| **[README.md](README.md)** | 项目导航和快速概览 | 所有用户 |
| **[UserGuide.md](UserGuide.md)** | 完整操作手册：安装、训练、评估、部署 | 学生、从业者 |
| **[DeveloperGuide.md](DeveloperGuide.md)** | 深度技术文档：架构、算法、源码解析 | 研究员、工程师 |

---

## 🏗️ 项目结构

```
x-humanoid-training-toolchain/
├── lerobot/                          # LeRobot 框架集成
│   ├── __init__.py
│   ├── configs/
│   │   ├── default.py
│   │   ├── train.py
│   │   ├── eval.py
│   │   ├── policies.py
│   │   ├── parser.py
│   │   └── types.py
│   ├── datasets/
│   │   ├── lerobot_dataset.py
│   │   ├── generate_metaworld_data.py
│   │   └── push_dataset_to_hub/
│   ├── common/
│   │   ├── datasets/                # 数据处理管道
│   │   ├── policies/                # 策略实现 (多种模型)
│   │   │   ├── act/                # ACT 模型
│   │   │   ├── diffusion/          # 扩散模型
│   │   │   ├── pi0/                # PI0 模型
│   │   │   ├── pi0fast/            # PI0-Fast 模型
│   │   │   ├── tdmpc/              # TDMPC 模型
│   │   │   ├── vqbet/              # VQ-BeT 模型
│   │   │   ├── factory.py
│   │   │   ├── normalize.py
│   │   │   ├── pretrained.py
│   │   │   └── utils.py
│   │   ├── robot_devices/           # 机器人硬件接口
│   │   ├── optim/                   # 优化器
│   │   ├── constants.py             # 常量定义
│   │   └── utils/                   # 通用工具函数
│   ├── scripts/
│   │   ├── train.py                 # 训练主脚本
│   │   ├── eval.py                  # 评估脚本
│   │   ├── lerobot_train.py
│   │   ├── lerobot_eval.py
│   │   ├── visualize_dataset.py
│   │   ├── visualize_dataset_html.py
│   │   ├── visualize_image_transforms.py
│   │   ├── control_robot.py         # 真机控制
│   │   ├── control_sim_robot.py     # 仿真控制
│   │   ├── configure_motor.py       # 电机配置
│   │   ├── find_motors_bus_port.py
│   │   ├── display_sys_info.py
│   │   └── push_pretrained.py       # 模型上传
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── configs.py
│   │   ├── factory.py
│   │   ├── libero.py
│   │   ├── metaworld.py
│   │   └── utils.py
│   └── README.md
├── scripts/                          # 项目特定脚本
│   ├── convert_to_lerobot.py        # HDF5 格式转换
│   ├── convert_to_lerobot_1.py
│   └── convert.sh
├── deployment/                       # 真机部署
│   ├── action_policy.py
│   ├── ros2_deployment_inspire.py
│   ├── ros2_deployment_brainco.py
│   └── readme.md
├── tests/                            # 单元测试
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_available.py
│   ├── utils.py
│   ├── artifacts/
│   ├── cameras/
│   ├── configs/
│   ├── datasets/
│   ├── envs/
│   ├── examples/
│   ├── fixtures/
│   ├── motors/
│   ├── optim/
│   ├── policies/
│   ├── robots/
│   └── utils/
├── examples/                         # 使用示例
│   ├── 1_load_lerobot_dataset.py
│   ├── 2_evaluate_pretrained_policy.py
│   ├── 3_train_policy.py
│   └── advanced/
│       ├── 1_add_image_transforms.py
│       └── 2_calculate_validation_loss.py
├── docker/
│   ├── lerobot-cpu/
│   ├── lerobot-gpu/
│   └── lerobot-gpu-dev/
├── benchmarks/
│   └── video/
├── media/                            # 媒体资源
├── patches/                          # 补丁文件
├── static/
├── train_example/                    # 训练示例
├── README.md                         # 项目首页
├── UserGuide.md                      # 用户手册
├── DeveloperGuide.md                 # 开发者指南
├── LICENSE
├── pyproject.toml
└── Makefile
```

---

## 🎓 支持的策略实现

VLA 框架集成了 LeRobot，支持多种机器人策略实现：

| 策略 | 说明 | 源码位置 |
|-----|------|---------|
| **ACT** | Action Chunking Transformer - 基于分块的变压器模型 | `lerobot/common/policies/act/` |
| **Diffusion** | 扩散模型策略 | `lerobot/common/policies/diffusion/` |
| **PI0** | 神经网络策略 | `lerobot/common/policies/pi0/` |
| **PI0-Fast** | PI0 的优化版本 | `lerobot/common/policies/pi0fast/` |
| **TDMPC** | Temporal Difference MPC | `lerobot/common/policies/tdmpc/` |
| **VQ-BeT** | VQ-VAE BeT 模型 | `lerobot/common/policies/vqbet/` |

详细的架构设计和实现细节请参考：
- 📖 [DeveloperGuide.md](DeveloperGuide.md) - 系统架构和核心模块解析
- 🔍 源码：`lerobot/common/policies/act/configuration_act.py` 和 `modeling_act.py`

---

## 🤖 真机部署

VLA 框架完全支持 Walker-TienKung 机器人的实时部署。

### 部署架构

```
┌─────────────────────┐
│   策略模型 (GPU)     │
│   ACT / VLA Model   │
└──────────┬──────────┘
           │ ROS2 话题
           ▼
┌──────────────────────┐
│  控制节点 (CPU/GPU)  │
│  ros2_deployment.py  │
└──────────┬───────────┘
           │ 命令
           ▼
┌──────────────────────┐
│   机器人中间件        │
│   - 运动学          │
│   - 碰撞检测        │
│   - 安全约束        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Walker-TienKung 硬件      │
└──────────────────────┘
```

## 🙏 致谢

本项目基于以下优秀开源项目：

- **[LeRobot](https://huggingface.co/docs/lerobot)** - Hugging Face 机器人学习框架
- **[ACT](https://github.com/tonyzhaozh/act)** - 动作分块变压器模型
- **[RoboMIND](https://huggingface.co/datasets/lerobot/robomind)** - 大规模行为克隆数据集
- **[ROS2 Humble](https://docs.ros.org/en/humble/)** - 机器人操作系统
- **[Diffusers](https://github.com/huggingface/diffusers)** - 扩散模型库

---

## 📄 许可证

本项目采用 [Apache 2.0 License](LICENSE) 开源。

---

## 📞 联系我们

- **GitHub Issues**: [报告问题或请求功能](https://github.com/your-org/x-humanoid-training-toolchain/issues)
- **讨论论坛**: [GitHub Discussions](https://github.com/your-org/x-humanoid-training-toolchain/discussions)

---

## 📌 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{vla_humanoid_2026,
  title={VLA: Vision-Language-Action Learning for Humanoid Robots},
  author={Your Team Name},
  year={2026},
  url={https://github.com/your-org/x-humanoid-training-toolchain}
}
```

---

## 🔗 相关资源

| 资源 | 链接 |
|-----|------|
| LeRobot 文档 | https://huggingface.co/docs/lerobot |
| ACT 论文 | https://arxiv.org/abs/2304.13705 |
| RoboMIND 数据 | https://huggingface.co/datasets/lerobot/robomind |
| ROS2 教程 | https://docs.ros.org/en/humble/Tutorials.html |
| PyTorch 官网 | https://pytorch.org |

---

<div align="center">
  <b>🚀 开始您的人形机器人模仿学习之旅!</b>
  <br><br>
  Made with ❤️ by ECBG Humanoid Team
</div>

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
