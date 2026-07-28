# x-humanoid-training-toolchain 模仿学习训练工具链 - 开发者指南

**版本**: v2.1.6
**最后更新**: 2026-02-24  
**框架**: LeRobot 0.3.4+ + 1105_2 数据集  

---

## 📖 目录

### 第一部分：架构概览
1. [系统架构](#1-系统架构)
2. [技术栈](#2-技术栈)
3. [代码组织](#3-代码组织)

### 第二部分：核心组件
4. [数据处理管道](#4-数据处理管道)
5. [策略模型架构](#5-策略模型架构)
6. [训练系统](#6-训练系统)
7. [推理与部署](#7-推理与部署)

### 第三部分：高级主题
8. [自定义任务开发](#8-自定义任务开发)
9. [性能优化](#9-性能优化)
10. [算法细节](#10-算法细节)

---

## 第一部分：架构概览

## 1. 系统架构

### 1.1 高层架构图

```
┌────────────────────────────────────────────────────────┐
│              用户脚本层                                 │
│  (train.py, eval.py, convert_to_lerobot_1.py)         │
└────────────────┬─────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────┐
│          LeRobot 框架层                               │
│  ┌──────────────────────────────────────────────┐   │
│  │ LeRobotDataset：数据加载和缓存               │   │
│  │ • 优化视频编码 (H.264)                       │   │
│  │ • 高效 episode 索引                          │   │
│  │ • 延迟加载与 RAM 缓冲                        │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │ ACT 策略模型：(Transformer + VAE)            │   │
│  │ • 视觉编码器 (ResNet/ViT)                    │   │
│  │ • Transformer 编码-解码器                    │   │
│  │ • VAE 隐变量分布                             │   │
│  │ • 动作解码器                                 │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │ 训练管道：                                    │   │
│  │ • 混合精度 (AMP)                             │   │
│  │ • 验证和检查点保存                            │   │
│  │ • WandB 日志记录                             │   │
│  └──────────────────────────────────────────────┘   │
└────────────────┬─────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────┐
│        PyTorch 和 CUDA 优化层                        │
│  • GPU 内存管理                                     │
│  • CUDA 内核加速                                   │
└────────────────────────────────────────────────────┘
```

### 1.2 数据流架构

```
1105_2 数据集 (HDF5)
        ↓
┌─────────────────────────────────┐
│   convert_to_lerobot_1.py       │
│  • 视频编码优化                 │
│  • Episode 索引                 │
│  • 元数据生成                   │
└─────────────────────────────────┘
        ↓
LeRobot 格式 (视频 + JSON)
        ↓
┌─────────────────────────────────┐
│   LeRobotDataset 加载器         │
│  • 视频解压                     │
│  • 帧采样                       │
│  • 观察值归一化                 │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│   ACT 策略训练                   │
│  • 动作分块                     │
│  • 行为克隆损失                 │
│  • 多任务学习                   │
└─────────────────────────────────┘
        ↓
训练好的策略模型 (PyTorch 检查点)
```

---

## 2. 技术栈

### 2.1 框架版本

| 组件 | 版本 | 用途 |
|:---|:---:|:---|
| **LeRobot** | 0.3.4+ | 核心训练框架 |
| **PyTorch** | 2.0+ | 深度学习引擎 |
| **Python** | 3.10+ | 编程语言 |
| **NVIDIA 驱动** | ≥ 535.x | GPU 支持 |
| **CUDA** | 12.1+ | GPU 计算（GPU驱动配套版本） |
| **FFmpeg** | ≥ 4.4 | 视频处理 |
| **HuggingFace** | 最新版 | 模型库和数据集 |

### 2.2 关键依赖

```
requirements-dev.txt:
├── torch (含 CUDA 支持)
├── transformers (视觉和语言模型)
├── diffusers (VAE 模型)
├── huggingface_hub (数据集管理)
├── h5py (HDF5 文件处理)
├── opencv-python (视频处理)
├── wandb (训练监控)
└── tqdm (进度条)
```

---

## 3. 代码组织

### 3.1 项目结构

```
x-humanoid-training-toolchain/
│
├── lerobot/                      # LeRobot 框架（集成）
│   ├── common/
│   │   ├── datasets/
│   │   │   └── lerobot_dataset.py  # 📌 核心数据集类
│   │   ├── policies/
│   │   │   ├── act/                # 📌 ACT 策略模型（目录）
│   │   │   │   ├── modeling_act.py # ACT 模型实现
│   │   │   │   └── configuration_act.py
│   │   │   ├── diffusion/          # 扩散策略
│   │   │   ├── pi0/                # PI0 策略
│   │   │   └── factory.py          # 策略工厂
│   │   ├── optim/
│   │   │   ├── factory.py          # 优化器工厂
│   │   │   └── schedulers.py       # 学习率调度器
│   │   └── utils/
│   │
│   └── scripts/
│       ├── train.py                # 📌 主训练入口
│       ├── eval.py                 # 📌 评估脚本
│       └── visualize_dataset.py    # 数据集检查
│
├── scripts/
│   ├── convert_to_lerobot_1.py    # 📌 HDF5 → LeRobot 转换
│   └── configs/
│       ├── train_config.json      # 训练超参数
│       ├── Tien_Kung_26_1RGB.json # 数据格式规范
│       └── all_robot_h5_info.md   # HDF5 字段定义
│
├── deployment/
│   ├── action_policy.py           # 📌 推理包装类
│   ├── ros2_deployment_inspire.py # ROS2 节点（灵巧手）
│   ├── ros2_deployment_brainco.py # ROS2 节点（BrainCo 手）
│   └── readme.md
│
└── examples/
    └── 1_load_lerobot_dataset.py  # 快速入门示例

📌 = 您可能需要修改的核心文件
```

---

## 第二部分：核心组件

## 4. 数据处理管道

### 4.1 理解 1105_2 数据集格式

**HDF5 结构：**

1105_2数据集采用特定的HDF5结构，主要包含以下键：

- `puppet/arm_left_position_align/data` - 左臂关节位置（7维）
- `puppet/arm_right_position_align/data` - 右臂关节位置（7维）
- `puppet/gripper_left_position_align/data` - 左手抓取器（6维）
- `puppet/gripper_right_position_align/data` - 右手抓取器（6维）
- `camera_observations/color_images/camera_head` - 头部相机图像（压缩格式）

**查看实际结构**：使用h5py库探索具体episode文件，或参考 `scripts/configs/Tien_Kung_26_1RGB.json` 中的字段定义。

### 4.2 转换管道 (HDF5 → LeRobot)

**`convert_to_lerobot_1.py` 核心流程：**

实际的数据转换使用 `process_episode()` 函数（非 `convert_episode`），主要步骤：

1. **读取 HDF5 文件** - 包含机器人关节位置、末端执行器位置和相机图像
2. **提取机械臂数据** - `puppet/arm_left_position_align/data`、`puppet/arm_right_position_align/data`
3. **处理图像数据** - `camera_observations/color_images/camera_head`（压缩格式）
4. **使用 LeRobotDataset API** - 通过 `dataset.add_frame()` 添加帧数据
5. **自动视频编码** - LeRobot 内部使用 H.264 编码压缩视频

**重要说明**：
- HDF5 数据结构为 1105_2 格式，不是通用的 `observations/images` 结构
- 实际函数签名：`process_episode(episode_path, dataset, task_name, image_size, args)`
- 视频编码由 LeRobotDataset 自动处理，不需手动调用 FFmpeg

**查看实际代码**：`scripts/convert_to_lerobot_1.py` 参见函数 process_episode() / ACT 类实现

**存储优化：**

- 原始 HDF5：每个 episode 可能为 GB 级别（取决于帧数和分辨率）
- 转换后 H.264：每个 episode 可能降至 MB 级别（取决于压缩参数）
- 总体数据规模取决于 episode 数量与压缩/存储设置；使用 H.264 压缩通常能显著降低存储需求。

### 4.3 LeRobotDataset 加载

**数据加载说明：**

LeRobotDataset 基于 HuggingFace datasets 实现，返回扁平化的字典结构（不是嵌套字典）。

**返回数据格式：**

```python
{
    "observation.images.camera_head": torch.Tensor,  # (3, H, W) 图像
    "observation.state": torch.Tensor,               # (26,) 关节状态
    "action": torch.Tensor,                          # (26,) 动作
    "action_is_pad": torch.Tensor,                   # (26,) 填充掩码
    "episode_index": int,                            # Episode 索引
    "task_index": int,                               # 任务索引
    "task": str,                                     # 任务名称
}
```

**重要特性：**

- **扁平化键名**：使用点号分隔（如 `observation.images.camera_head`）
- **视频延迟加载**：视频帧按需解码，节省内存
- **自动归一化**：图像和动作在策略内部归一化
- **填充处理**：`action_is_pad` 标记填充帧，用于损失计算

**查看实际实现**：`lerobot/common/datasets/lerobot_dataset.py`

---

## 5. 策略模型架构

### 5.1 ACT (Action Chunking with Transformers) 概览

**模型架构：**

```
输入图像序列 (T 帧)
        ↓
┌─────────────────────────────┐
│   视觉编码器                │
│  (ResNet-50 或 ViT)         │
│  输入：(T, 3, H, W)        │
│  输出：(T, D) 特征         │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│  Transformer 编码器         │
│  • 多头注意力               │
│  • 前馈层                   │
│  • 层归一化                 │
│  (T, D) → (T, D)           │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│  Transformer 解码器         │
│  • 交叉注意力到             │
│    编码器输出               │
│  • 自回归或                 │
│    并行预测                 │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│  动作解码器 (VAE)            │
│  • 变分编码器：             │
│    均值与方差               │
│  • 重参数化                 │
│  • 动作重构                 │
│  输出：(A, 26)             │
│  其中 A = action_chunk_sz   │
└─────────────────────────────┘
```

### 5.2 VAE（变分自编码器）组件

**为什么使用 VAE？**

- 捕捉动作分布的多模态性
- 允许随机动作采样
- 规范化策略行为

**实际实现说明：**

ACT 模型中的 VAE 实现集成在 Transformer 架构中，并非独立类。核心实现在 `lerobot/common/policies/act/modeling_act.py` 文件的 `ACT` 类中：

- **VAE 编码器**：使用 `self.vae_encoder`（基于 Transformer 的 ACTEncoder）
- **潜在空间投影**：通过 `self.vae_encoder_latent_output_proj` 生成均值和方差
- **重参数化**：在 forward 方法中实现采样
- **KL 散度损失**：在 `ACTPolicy.forward()` 中计算

**查看实际代码**：请参考 `lerobot/common/policies/act/modeling_act.py` 第302-331行的VAE实现。

### 5.3 动作分块

**概念：**

与其预测每帧一个动作，不如一次预测一个"块"的 A 个动作。

```
帧序列：
[Frame_t, Frame_t+1, ..., Frame_t+T]
        ↓
[Action_t, Action_t+1, ..., Action_t+A-1]  ← 动作块
        ↓
在机器人上部署：
执行第一个动作，移动到下一帧
重复预测...
```

**优点：**
- 减少预测频率（更稳定）
- 捕捉动作相关性
- 提升长期性能

---

## 6. 训练系统

### 6.1 训练流程

训练系统的核心实现在 `lerobot/scripts/train.py` 文件中。

**主要组件**：
- **update_policy()** 函数：执行单步训练更新（梯度计算、裁剪、优化）
- **混合精度训练**：使用 `torch.cuda.amp.GradScaler` 和 `autocast`
- **梯度裁剪**：通过 `torch.nn.utils.clip_grad_norm_` 防止梯度爆炸
- **学习率调度**：支持配置学习率调度器

**训练配置**：
- 使用 JSON 配置文件（如 `scripts/configs/train_config.json`）
- 支持 WandB 日志记录
- 定期保存检查点（checkpoint）
- 自动验证和评估

**查看实际实现**：`lerobot/scripts/train.py` 第124-168行（update_policy函数）

### 6.2 损失函数

**行为克隆损失：**

```
L_BC = L1(预测动作, 真实动作)
```

**说明**：ACT 模型使用 L1 损失（平均绝对误差）而非 MSE，因为 L1 损失对异常值更鲁棒。

**VAE 损失（若适用）：**

```
L_VAE = L_重构 + β * L_KL
      = L1(动作, 重构动作) + KL_散度
```

**多任务损失（若训练多个任务）：**

```
L_总 = Σ_任务 (权重_任务 * L_任务)
```

---

## 7. 推理与部署

### 7.1 PolicyAgent 类

**核心推理包装器 (`action_policy.py`)：**

PolicyAgent 类的主要功能和接口：

```python
class PolicyAgent:
    def __init__(self, model_path):
        """
        加载训练好的策略模型

        参数：
            model_path: 预训练模型路径

        注意：设备自动检测，无需手动指定
        """
        self.model_path = Path(model_path)
        self.device = self._get_device()  # 自动检测 CUDA
        self.policy = self._load_policy()  # 加载策略
        self.cnt = 0  # 推理计数器

    def inference(self, obs):
        """
        执行模型推理

        参数：
            obs: 包含 'images' 字典的观察数据
                 obs['images']['camera_head'] = numpy array (H, W, 3)

        返回：
            action: numpy array (26,)
        """
        # 准备输入数据
        input_data = self.prepare_inference_obs(obs)

        # 调用策略的 select_action 方法
        return self.policy.select_action(input_data)

    def prepare_inference_obs(self, obs):
        """
        处理观察数据为模型输入格式

        返回：扁平化字典，key 为 'observation.images.camera_head'
        """
        # 图像预处理：resize、归一化、转 tensor
        # 详见 deployment/action_policy.py 实现
```

**重要说明**：
- 方法名是 `inference()`，不是 `forward()`
- 观察数据结构：`obs['images']['camera_head']`（嵌套字典）
- 返回的输入数据使用扁平化 key：`'observation.images.camera_head'`
- 调用 `policy.select_action()` 而非直接调用模型

**查看实际代码**：`deployment/action_policy.py`

### 7.2 ROS2 集成

**ROS2 部署实现：**

项目提供了完整的ROS2部署节点实现，支持双臂双手控制：

- **Inspire灵巧手版本**：`deployment/ros2_deployment_inspire.py`
- **BrainCo手版本**：`deployment/ros2_deployment_brainco.py`

**关键实现特性**：

1. **消息同步**：使用 `message_filters.ApproximateTimeSynchronizer` 同步多路图像和关节状态
2. **策略推理**：通过 `PolicyAgent.inference()` 执行模型推理
3. **双臂控制**：发布到 `/inspire/left_hand_position` 和 `/inspire/right_hand_position` 话题
4. **安全机制**：包含关节限位检查和错误处理

**部署步骤**：

```bash
# 1. 编辑部署脚本，设置模型路径（第42行）
# 2. 启动ROS2节点
python3 deployment/ros2_deployment_inspire.py

# 3. 验证节点运行
ros2 node list
ros2 topic list
```

**查看完整实现**：`deployment/ros2_deployment_inspire.py` 和 `deployment/readme.md`

---

## 第三部分：高级主题

## 8. 自定义任务开发

### 8.1 自定义数据格式

如果您有不同的数据结构（不是 1105_2），需要创建自己的转换脚本。

**推荐方法**：使用 LeRobot的Dataset API

1. **创建 LeRobotDataset** - 使用 `LeRobotDataset.create()`
2. **添加帧数据** - 使用 `dataset.add_frame()` 方法
3. **整合episode** - 使用 `dataset.consolidate()` 完成数据集

**参考实现**：
- `scripts/convert_to_lerobot_1.py` - 1105_2格式转换示例
- `lerobot/common/datasets/lerobot_dataset.py` - Dataset API文档

**关键步骤**：
- 读取你的数据格式（HDF5, JSON, 或其他）
- 提取图像和动作数据
- 使用LeRobotDataset API逐帧添加
- 视频编码由LeRobot自动处理

---

## 9. 性能优化

### 9.1 训练速度优化

**已实现的优化**：

1. **混合精度训练（AMP）**：
   - 实现位置：`lerobot/scripts/train.py`
   - 使用 `torch.cuda.amp.GradScaler` 和 `autocast`
   - 配置参数：`policy.use_amp=true`

2. **数据加载优化**：
   - 多线程加载：配置 `num_workers` 参数
   - 固定内存：自动启用 `pin_memory`
   - 配置位置：训练配置文件

3. **梯度裁剪**：
   - 防止梯度爆炸
   - 实现位置：`update_policy()` 函数

### 9.2 内存优化

**实际使用的优化**：

1. **检查点保存**：
   - 定期保存模型检查点
   - 配置参数：`save_freq`（默认每10000步保存一次）
   - 保存位置：`output_dir/checkpoints/`

2. **批大小调整**：
   - 减小 `batch_size` 降低显存占用
   - 减小 `policy.chunk_size` 降低序列长度

**查看实际实现**：`lerobot/scripts/train.py`

---

## 10. 算法细节

### 10.1 行为克隆理论

**目标：**

```
min_θ E[||π_θ(a_t | o_t) - a_expert(o_t)||²]
```

其中：
- π_θ = 由 θ 参数化的学习策略
- a_t = 动作
- o_t = 观察
- a_expert = 专家（人类）动作

### 10.2 分布不匹配问题

**挑战**：当策略出错时，它遇到的观察与训练数据不同。

**本项目中的解决方案：**

1. **动作分块**：减少预测频率
2. **Transformer**：使用长上下文注意力处理变异
3. **VAE**：建模动作分布，而非点估计

### 10.3 多任务学习

**共享表示策略：**

ACT策略通过共享视觉编码器实现多任务学习：

- **共享层**：ResNet或ViT视觉编码器对所有任务通用
- **任务条件**：通过 `task_index` 和 `task_embedding` 区分不同任务
- **损失计算**：训练时根据batch中的 `task_index` 计算对应任务的L1损失

**实际实现**：
- ACT模型架构：`lerobot/common/policies/act/modeling_act.py`
- 多任务训练：数据集自动提供 `task_index` 字段
- 任务权重：可通过训练配置调整不同任务的采样比例

---

## 常见开发任务

### 数据集验证

```bash
# 使用LeRobot提供的可视化工具
python lerobot/scripts/visualize_dataset.py \
    dataset.repo_id=your-dataset

# 或使用HTML可视化
python lerobot/scripts/visualize_dataset_html.py \
    dataset.repo_id=your-dataset
```

### 训练监控

```bash
# 训练会自动记录到WandB
# 访问 https://wandb.ai 查看实时训练指标

# 检查训练输出目录
ls outputs/train/*/checkpoints/
```

### 模型评估

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

# 参考示例代码
python examples/2_evaluate_pretrained_policy.py
```

**查看更多示例**：`examples/` 目录包含完整的使用示例代码

---

## 参考文献

- **ACT 论文**：Zhao et al., "Action Chunking with Transformers", 2023
- **行为克隆**：Bain & Sammut, "A Framework for Behavioural Cloning", 1999
- **Transformers**：Vaswani et al., "Attention is All You Need", 2017
- **VAE**：Kingma & Welling, "Auto-Encoding Variational Bayes", 2013
- **LeRobot 文档**：https://github.com/huggingface/lerobot

---

<div align="center">
  <b>🚀 开始您的人形机器人模仿学习之旅!</b>
  <br><br>
  Made with ❤️ by ECBG Humanoid Team
</div>