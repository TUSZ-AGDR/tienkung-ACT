# 评估脚本：基于 ACT 模型的单帧预测能力验证，本质就是开卷考试。平均误差要在 0.007以下才算通过。
import torch,tqdm
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.act.modeling_act import ACTPolicy

def validate():
    # 1. 基础配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    #替换模型路径及数据集路径
    pretrained_model_path = "/home/ubuntu/x-humanoid-training-toolchain_20260224v2.1.6/x-humanoid-training-toolchain/data/output/pick_place/checkpoints/last/pretrained_model"
    dataset_path = "/home/ubuntu/x-humanoid-training-toolchain_20260224v2.1.6/x-humanoid-training-toolchain/data/dataset/pick_place"
    
    # 2. 加载模型
    print(f"加载模型: {pretrained_model_path}")
    print(f"数据集: {dataset_path}")
    policy = ACTPolicy.from_pretrained(pretrained_model_path).to(device).eval()

    # 3. 准备数据集配置
    # 获取元数据以读取 FPS 和总集数
    meta = LeRobotDatasetMetadata(dataset_path)
    
    # 动态构建 delta_timestamps (适配 ACT chunking)
    # 这里的 chunk_size 直接从策略配置读取，fps 从数据集读取
    delta_timestamps = {
        "observation.images.camera_head": [0.0],
        "observation.state": [0.0],
        "action": [i / meta.fps for i in range(policy.config.chunk_size)],
    }

    # 4. 加载验证集
    # 注意：LeRobotDataset 的 episodes 参数在配合 delta_timestamps 使用时可能存在索引不对齐的 bug，
    # 导致 IndexError。因此这里加载完整数据集，通过 Subset 提取验证集帧范围。
    full_dataset = LeRobotDataset(dataset_path, delta_timestamps=delta_timestamps)
    
    # 确定验证集的帧范围：随机抽取所有帧的 5%
    total_frames = len(full_dataset)
    num_val_frames = int(total_frames * 0.05)
    if num_val_frames < 1: num_val_frames = 1
    
    # 随机生成帧索引
    # 使用固定种子保证每次运行选取的验证集一致（如果需要完全随机可去掉 manual_seed）
    # torch.manual_seed(42) 
    val_indices = torch.randperm(total_frames)[:num_val_frames].tolist()

    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    print(f"验证集: 随机选择了 {len(val_dataset)}/{total_frames} 帧 (5%)")

    # 使用 DataLoader 自动处理 batching 和数据整理
    # shuffle=True 以便随机采样验证集中的帧，而不是只验证开头的那些帧
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

    # 5. 推理循环
    l1_errors = []
    print("\n开始推理...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(dataloader, desc="验证中", unit="batch")):
            
            # 自动迁移所有 Tensor 到设备
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() 
                    if isinstance(v, torch.Tensor)}
            # 1. 移除 'action'，防止 ACT 使用 VAE 编码器读取 GT 动作 (Leakage)。
            #    推理时应当基于 observation 生成 latent (或者使用 zero latent)。
            input_batch = {k: v for k, v in batch.items() if k != "action"}

            # 2. 压缩 Observation 的时间维度
            #    Dataset 设置 delta_timestamps=[0.0] 会返回 (B, 1, ...) 格式，
            #    但 ACT 模型期望 Observation 是 (B, ...) 格式 (单帧输入)。
            for key in input_batch:
                if key.startswith("observation.") and input_batch[key].ndim > 2:
                    # 对于图像 (B, 1, C, H, W) -> (B, C, H, W)
                    # 对于状态 (B, 1, D) -> (B, D)
                    if input_batch[key].shape[1] == 1:
                        input_batch[key] = input_batch[key].squeeze(1)

            # 重要：ACT 是有状态策略 (Chunking/Temporal Ensemble)。
            # 若验证"每一帧基于当前观测的独立预测能力"，必须每次重置，防止使用历史缓存动作。
            policy.reset()



            # 预测
            # 使用处理过的 input_batch
            actions = policy.select_action(input_batch)
            # 对比当前时刻 (t=0) 的 GT 动作
            # batch['action'] 也是 (B, Chunk, D)
            gt_action = batch["action"][:, 0]
            
            # 兼容处理: 如果输出是序列则取第一个，如果是单步则直接使用
            pred_action = actions[:, 0] if actions.ndim == 3 else actions

            # 计算误差
            error = torch.mean(torch.abs(pred_action - gt_action)).item()
            l1_errors.append(error)
            # 调试信息 (仅首帧)
            if i == 0: 
                # 此时 input_batch["observation.images.camera_head"] 已经是 (B, C, H, W)
                img = input_batch["observation.images.camera_head"]
                torchvision.utils.save_image(img[0], "debug_observation.png")
                print(f"Debug image saved. Input shape: {img.shape}")
                # 打印样例值 (保留3位小数)
                def format_tensor(tensor):
                    data = tensor.detach().cpu().numpy().tolist()
                    return ','.join([f"{x:.3f}" for x in data])
                print(f"[{i:02d}] \nPred: {format_tensor(pred_action[0])} \n GT : {format_tensor(gt_action[0])}")

    if l1_errors:
        print(f"\n验证完成！平均误差 (L1): {np.mean(l1_errors):.6f}")

if __name__ == "__main__":
    validate()
