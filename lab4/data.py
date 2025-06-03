# lab4/data.py  – 数据预处理与数据增强模块
# 实现数据清洗、特征提取、归一化和数据增强功能

import os
import numpy as np
import torch
import random
from typing import List, Tuple, Dict, Optional
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings("ignore")


# -----------------------------------------------------------
# 关键关节索引定义（根据PDF要求）
# -----------------------------------------------------------
# 根据人体结构，提取头部、躯干、四肢等关键关节
# 注意：这些索引会根据实际数据维度进行动态调整

def get_key_joint_indices(data_dim: int) -> List[int]:
    """
    根据数据维度动态生成关键关节索引
    
    Args:
        data_dim: 数据特征维度
        
    Returns:
        关键关节索引列表
    """
    # 基础关键关节模式（每个关节3个坐标：X, Y, Z）
    base_joints = {
        # 躯干核心 (前几个关节通常是躯干)
        'chest': [4, 5, 6],        # 胸部
        'neck': [7, 8, 9],         # 颈部  
        'head': [10, 11, 12],      # 头部
        
        # 上肢关节
        'l_shoulder': [13, 14, 15], # 左肩
        'l_elbow': [16, 17, 18],    # 左肘
        'l_wrist': [19, 20, 21],    # 左腕
        
        'r_shoulder': [22, 23, 24], # 右肩
        'r_elbow': [25, 26, 27],    # 右肘
        'r_wrist': [28, 29, 30],    # 右腕
        
        # 下肢关节
        'l_hip': [31, 32, 33],      # 左髋
        'l_knee': [34, 35, 36],     # 左膝
        'l_ankle': [37, 38, 39],    # 左踝
        
        'r_hip': [40, 41, 42],      # 右髋
        'r_knee': [43, 44, 45],     # 右膝
        'r_ankle': [46, 47, 48],    # 右踝
    }
    
    # 收集所有在数据维度范围内的索引
    key_indices = []
    for joint_name, indices in base_joints.items():
        for idx in indices:
            if idx < data_dim:
                key_indices.append(idx)
    
    # 如果没有找到合适的关键关节，使用前面的一些索引
    if not key_indices:
        key_indices = list(range(min(24, data_dim)))  # 取前24个特征作为关键特征
    
    return sorted(key_indices)

# 默认关键关节索引（会在运行时动态更新）
KEY_JOINT_INDICES = list(range(24))  # 默认前24个特征


# -----------------------------------------------------------
# 数据清洗与预处理
# -----------------------------------------------------------
def clean_skeleton_data(data: np.ndarray, 
                       zero_threshold: float = 1e-6,
                       outlier_threshold: float = 5000.0) -> np.ndarray:
    """
    数据清洗功能
    
    Args:
        data: 原始骨骼数据 (frames, features)
        zero_threshold: 零值阈值
        outlier_threshold: 异常值阈值
    
    Returns:
        清洗后的数据
    """
    cleaned_data = data.copy()
    
    # 1. 过滤无效帧：剔除关节坐标全为零的时间帧
    valid_frames = []
    for i, frame in enumerate(cleaned_data):
        # 检查关键关节是否全为零
        key_joints = frame[KEY_JOINT_INDICES] if len(KEY_JOINT_INDICES) <= len(frame) else frame
        if not np.all(np.abs(key_joints) < zero_threshold):
            valid_frames.append(i)
    
    if len(valid_frames) == 0:
        raise ValueError("所有帧的关键关节都为零值，无法处理")
    
    cleaned_data = cleaned_data[valid_frames]
    
    # 2. 异常值处理：剔除明显异常的坐标值
    for i in range(cleaned_data.shape[1]):
        col = cleaned_data[:, i]
        # 使用IQR方法检测异常值
        q75, q25 = np.percentile(col, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        # 将异常值替换为中位数
        median_val = np.median(col)
        mask = (col < lower_bound) | (col > upper_bound) | (np.abs(col) > outlier_threshold)
        cleaned_data[mask, i] = median_val
    
    # 3. 缺失值处理：使用线性插值
    for i in range(cleaned_data.shape[1]):
        col = cleaned_data[:, i]
        if np.any(np.isnan(col)) or np.any(np.isinf(col)):
            # 使用前后值的平均数填充
            valid_indices = ~(np.isnan(col) | np.isinf(col))
            if np.any(valid_indices):
                cleaned_data[:, i] = np.interp(
                    np.arange(len(col)), 
                    np.where(valid_indices)[0], 
                    col[valid_indices]
                )
            else:
                cleaned_data[:, i] = 0
    
    return cleaned_data


def extract_key_joints(data: np.ndarray) -> np.ndarray:
    """
    提取关键关节坐标
    
    Args:
        data: 完整骨骼数据
    
    Returns:
        关键关节数据
    """
    if data.shape[1] < max(KEY_JOINT_INDICES):
        # 如果数据维度不足，返回原数据
        return data
    
    return data[:, KEY_JOINT_INDICES]


def body_centered_normalization(data: np.ndarray, 
                               chest_indices: List[int] = [0, 1, 2]) -> np.ndarray:
    """
    以胸腔关节为中心的坐标归一化
    
    Args:
        data: 关节数据 (frames, joints*3)
        chest_indices: 胸腔关节在关键关节中的索引
    
    Returns:
        归一化后的数据
    """
    normalized_data = data.copy()
    
    # 确保胸腔索引存在
    if max(chest_indices) >= data.shape[1]:
        # 如果没有胸腔数据，使用第一个关节作为参考
        chest_indices = [0, 1, 2] if data.shape[1] >= 3 else [0]
    
    for frame_idx in range(data.shape[0]):
        frame = normalized_data[frame_idx]
        
        # 获取胸腔中心坐标
        if len(chest_indices) >= 3:
            chest_center = frame[chest_indices]
        else:
            chest_center = np.array([frame[chest_indices[0]], 0, 0])
            if len(chest_indices) >= 2:
                chest_center[1] = frame[chest_indices[1]]
            if len(chest_indices) >= 3:
                chest_center[2] = frame[chest_indices[2]]
        
        # 将所有关节坐标转换为相对于胸腔的坐标
        for joint_idx in range(0, min(len(frame), len(KEY_JOINT_INDICES)), 3):
            if joint_idx + 2 < len(frame):
                frame[joint_idx:joint_idx+3] -= chest_center
        
        normalized_data[frame_idx] = frame
    
    return normalized_data


def temporal_sampling(data: np.ndarray, sampling_interval: int = 4) -> np.ndarray:
    """
    时序采样：按固定间隔采样，降低数据维度
    
    Args:
        data: 输入数据 (frames, features)
        sampling_interval: 采样间隔
    
    Returns:
        采样后的数据
    """
    # 从第3帧开始，每4帧采样一次（如PDF中的逻辑）
    start_frame = 3
    sampled_indices = list(range(start_frame, data.shape[0], sampling_interval))
    
    if len(sampled_indices) == 0:
        # 如果没有采样到任何帧，至少保留第一帧
        sampled_indices = [0]
    
    return data[sampled_indices]


# -----------------------------------------------------------
# 数据增强功能
# -----------------------------------------------------------
class SkeletonDataAugmenter:
    """骨骼数据增强器"""
    
    def __init__(self, 
                 rotation_range: float = 15.0,      # 旋转角度范围（度）
                 noise_std: float = 0.02,           # 噪声标准差
                 time_warp_ratio: float = 0.1):     # 时间扭曲比例
        self.rotation_range = rotation_range
        self.noise_std = noise_std
        self.time_warp_ratio = time_warp_ratio
    
    def random_rotation(self, data: np.ndarray) -> np.ndarray:
        """
        随机旋转：对关节坐标施加小幅度随机旋转，模拟不同视角
        
        Args:
            data: 输入数据 (frames, joints*3)
        
        Returns:
            旋转后的数据
        """
        augmented_data = data.copy()
        
        # 生成随机旋转角度
        rotation_angles = np.random.uniform(
            -self.rotation_range, 
            self.rotation_range, 
            3
        ) * np.pi / 180.0  # 转换为弧度
        
        # 创建旋转矩阵
        rotation_matrix = R.from_euler('xyz', rotation_angles).as_matrix()
        
        # 对每一帧的所有关节进行旋转
        for frame_idx in range(augmented_data.shape[0]):
            frame = augmented_data[frame_idx]
            
            # 将关节坐标重组为 (n_joints, 3) 的形式
            n_coords = len(frame) // 3 * 3  # 确保是3的倍数
            if n_coords > 0:
                joints = frame[:n_coords].reshape(-1, 3)
                
                # 应用旋转
                rotated_joints = joints @ rotation_matrix.T
                
                # 重新展平
                augmented_data[frame_idx, :n_coords] = rotated_joints.flatten()
        
        return augmented_data
    
    def add_gaussian_noise(self, data: np.ndarray) -> np.ndarray:
        """
        添加高斯噪声：在坐标中注入高斯噪声，提升模型鲁棒性
        
        Args:
            data: 输入数据
        
        Returns:
            添加噪声后的数据
        """
        noise = np.random.normal(0, self.noise_std, data.shape)
        return data + noise
    
    def time_warping(self, data: np.ndarray) -> np.ndarray:
        """
        时间扭曲：随机删除或重复部分帧，模拟动作速度变化
        
        Args:
            data: 输入数据 (frames, features)
        
        Returns:
            时间扭曲后的数据
        """
        n_frames = data.shape[0]
        
        # 计算要扭曲的帧数
        n_warp = max(1, int(n_frames * self.time_warp_ratio))
        
        # 随机选择扭曲类型：删除或重复
        if random.random() > 0.5:
            # 删除帧
            if n_frames > n_warp + 1:  # 确保至少保留一些帧
                indices_to_remove = random.sample(range(n_frames), n_warp)
                remaining_indices = [i for i in range(n_frames) if i not in indices_to_remove]
                return data[remaining_indices]
        else:
            # 重复帧
            indices_to_repeat = random.sample(range(n_frames), n_warp)
            augmented_data = [data]
            for idx in indices_to_repeat:
                augmented_data.append(data[idx:idx+1])
            return np.vstack(augmented_data)
        
        return data
    
    def random_scale(self, data: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        随机缩放：对整体动作进行缩放
        
        Args:
            data: 输入数据
            scale_range: 缩放范围
        
        Returns:
            缩放后的数据
        """
        scale_factor = random.uniform(scale_range[0], scale_range[1])
        return data * scale_factor
    
    def random_translation(self, data: np.ndarray, translation_range: float = 0.1) -> np.ndarray:
        """
        随机平移：对整体动作进行平移
        
        Args:
            data: 输入数据
            translation_range: 平移范围
        
        Returns:
            平移后的数据
        """
        translation = np.random.uniform(-translation_range, translation_range, 3)
        augmented_data = data.copy()
        
        # 对每一帧的所有关节进行平移
        for frame_idx in range(augmented_data.shape[0]):
            frame = augmented_data[frame_idx]
            n_coords = len(frame) // 3 * 3
            if n_coords > 0:
                joints = frame[:n_coords].reshape(-1, 3)
                joints += translation
                augmented_data[frame_idx, :n_coords] = joints.flatten()
        
        return augmented_data
    
    def augment(self, data: np.ndarray, 
                augment_types: List[str] = ['rotation', 'noise']) -> np.ndarray:
        """
        综合数据增强
        
        Args:
            data: 输入数据
            augment_types: 增强类型列表
        
        Returns:
            增强后的数据
        """
        augmented_data = data.copy()
        
        for aug_type in augment_types:
            if aug_type == 'rotation':
                augmented_data = self.random_rotation(augmented_data)
            elif aug_type == 'noise':
                augmented_data = self.add_gaussian_noise(augmented_data)
            elif aug_type == 'time_warp':
                augmented_data = self.time_warping(augmented_data)
            elif aug_type == 'scale':
                augmented_data = self.random_scale(augmented_data)
            elif aug_type == 'translation':
                augmented_data = self.random_translation(augmented_data)
        
        return augmented_data


# -----------------------------------------------------------
# 数据预处理流水线
# -----------------------------------------------------------
def preprocess_skeleton_data(raw_data: np.ndarray,
                           extract_keys: bool = True,
                           normalize: bool = True,
                           temporal_sample: bool = True,
                           sampling_interval: int = 4) -> np.ndarray:
    """
    完整的骨骼数据预处理流水线
    
    Args:
        raw_data: 原始数据
        extract_keys: 是否提取关键关节
        normalize: 是否进行归一化
        temporal_sample: 是否进行时序采样
        sampling_interval: 采样间隔
    
    Returns:
        预处理后的数据
    """
    # 1. 数据清洗
    cleaned_data = clean_skeleton_data(raw_data)
    
    # 2. 特征提取（关键关节）
    if extract_keys:
        processed_data = extract_key_joints(cleaned_data)
    else:
        processed_data = cleaned_data
    
    # 3. 坐标归一化
    if normalize:
        processed_data = body_centered_normalization(processed_data)
    
    # 4. 时序采样
    if temporal_sample:
        processed_data = temporal_sampling(processed_data, sampling_interval)
    
    return processed_data


# -----------------------------------------------------------
# 增强数据集类
# -----------------------------------------------------------
class AugmentedSkeletonDataset(torch.utils.data.Dataset):
    """
    支持数据增强的骨骼数据集
    """
    
    def __init__(self, file_paths: List[str], 
                 augment_prob: float = 0.5,
                 augment_types: List[str] = ['rotation', 'noise'],
                 preprocess_params: Optional[Dict] = None):
        """
        Args:
            file_paths: 数据文件路径列表
            augment_prob: 数据增强概率
            augment_types: 数据增强类型
            preprocess_params: 预处理参数
        """
        super().__init__()
        
        self.augment_prob = augment_prob
        self.augmenter = SkeletonDataAugmenter()
        self.augment_types = augment_types
        
        # 预处理参数
        self.preprocess_params = preprocess_params or {
            'extract_keys': True,
            'normalize': True,
            'temporal_sample': True,
            'sampling_interval': 4
        }
        
        # 加载和预处理数据
        self._load_data(file_paths)
    
    def _load_data(self, file_paths: List[str]):
        """加载数据"""
        from utils import _read_single_csv, _extract_action_id
        
        arrays, labels = [], []
        
        for path in file_paths:
            try:
                # 提取标签
                label = _extract_action_id(os.path.basename(path))
                if label == -1:
                    continue
                
                # 读取和预处理数据
                raw_data = _read_single_csv(path)
                processed_data = preprocess_skeleton_data(raw_data, **self.preprocess_params)
                
                arrays.append(processed_data)
                labels.append(label)
                
            except Exception as e:
                print(f"警告：处理文件 {path} 时出错: {e}")
                continue
        
        if not arrays:
            raise RuntimeError("没有成功加载任何数据文件")
        
        # 统一数据维度
        self.max_len = max(a.shape[0] for a in arrays)
        self.max_dim = max(a.shape[1] for a in arrays)
        
        self.samples, self.labels = [], []
        
        for data, label in zip(arrays, labels):
            # 帧长度padding
            if data.shape[0] < self.max_len:
                pad_frames = np.zeros((self.max_len - data.shape[0], data.shape[1]), dtype=np.float32)
                data = np.vstack([data, pad_frames])
            
            # 特征维度padding
            if data.shape[1] < self.max_dim:
                pad_features = np.zeros((self.max_len, self.max_dim - data.shape[1]), dtype=np.float32)
                data = np.hstack([data, pad_features])
            
            self.samples.append(torch.tensor(data, dtype=torch.float32))
            self.labels.append(torch.tensor(label, dtype=torch.long))
        
        self.feature_dim = self.max_dim
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx].clone()
        label = self.labels[idx]
        
        # 随机决定是否进行数据增强
        if random.random() < self.augment_prob:
            sample_np = sample.numpy()
            augmented_np = self.augmenter.augment(sample_np, self.augment_types)
            sample = torch.tensor(augmented_np, dtype=torch.float32)
            
            # 确保维度一致
            if sample.shape != self.samples[idx].shape:
                # 如果时间扭曲改变了帧数，需要重新padding或截断
                if sample.shape[0] > self.max_len:
                    sample = sample[:self.max_len]
                elif sample.shape[0] < self.max_len:
                    pad_frames = torch.zeros(self.max_len - sample.shape[0], sample.shape[1])
                    sample = torch.cat([sample, pad_frames], dim=0)
        
        return sample, label


# -----------------------------------------------------------
# 工具函数
# -----------------------------------------------------------
def create_augmented_dataset(data_dir: str,
                           train_ratio: float = 0.8,
                           augment_prob: float = 0.5,
                           augment_types: List[str] = ['rotation', 'noise'],
                           seed: Optional[int] = None) -> Tuple[AugmentedSkeletonDataset, AugmentedSkeletonDataset]:
    """
    创建训练和测试的增强数据集
    
    Args:
        data_dir: 数据目录
        train_ratio: 训练集比例
        augment_prob: 数据增强概率
        augment_types: 数据增强类型
        seed: 随机种子
    
    Returns:
        (训练集, 测试集)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 收集所有CSV文件
    csv_files = [os.path.join(data_dir, f) 
                 for f in os.listdir(data_dir) 
                 if f.lower().endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"在目录 {data_dir} 中未找到CSV文件")
    
    # 随机划分训练和测试集
    random.shuffle(csv_files)
    split_idx = int(len(csv_files) * train_ratio)
    train_files = csv_files[:split_idx]
    test_files = csv_files[split_idx:]
    
    # 创建数据集
    train_dataset = AugmentedSkeletonDataset(
        train_files, 
        augment_prob=augment_prob,
        augment_types=augment_types
    )
    
    test_dataset = AugmentedSkeletonDataset(
        test_files,
        augment_prob=0.0,  # 测试集不进行数据增强
        augment_types=[]
    )
    
    return train_dataset, test_dataset


def visualize_augmentation_effect(data: np.ndarray, 
                                augment_types: List[str] = ['rotation', 'noise']) -> Dict[str, np.ndarray]:
    """
    可视化数据增强效果
    
    Args:
        data: 原始数据
        augment_types: 增强类型
    
    Returns:
        包含原始数据和各种增强结果的字典
    """
    augmenter = SkeletonDataAugmenter()
    results = {'original': data}
    
    for aug_type in augment_types:
        augmented = augmenter.augment(data, [aug_type])
        results[aug_type] = augmented
    
    return results


if __name__ == "__main__":
    # 测试代码
    print("骨骼数据预处理与增强模块测试")
    
    # 测试数据预处理
    dummy_data = np.random.randn(100, 60)  # 100帧，60个特征
    processed = preprocess_skeleton_data(dummy_data)
    print(f"原始数据形状: {dummy_data.shape}")
    print(f"处理后数据形状: {processed.shape}")
    
    # 测试数据增强
    augmenter = SkeletonDataAugmenter()
    augmented = augmenter.augment(processed, ['rotation', 'noise'])
    print(f"增强后数据形状: {augmented.shape}")
    
    print("测试完成！")
