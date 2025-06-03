# lab4/skeleton_utils.py - 改进的骨架数据处理与增强模块
import os, re
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import random

# -----------------------------------------------------------
# 骨架CSV文件解析 - 针对实际文件结构优化
# -----------------------------------------------------------
class SkeletonCSVParser:
    """解析复杂格式的骨架CSV文件"""
    
    def __init__(self):
        self.metadata = {}
        self.joint_names = []
        self.joint_types = []
        self.data_types = []
        
    def parse_csv(self, filepath: str) -> Tuple[np.ndarray, Dict]:
        """
        解析骨架CSV文件，返回数据矩阵和元数据
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 解析元数据（第一行）
        self._parse_metadata(lines[0])
        
        # 解析类型行（第二行）
        self._parse_types(lines[1])
        
        # 解析关节名称（第三行）
        self._parse_names(lines[2])
        
        # 解析ID行（第四行） - 跳过
        
        # 解析数据类型行（第五行）
        self._parse_data_types(lines[4])
        
        # 解析列标题行（第六行） - 跳过
        
        # 解析数据行（从第七行开始）
        data_rows = []
        for line in lines[6:]:
            line = line.strip()
            if not line:
                continue
            
            values = line.split(',')
            try:
                # 跳过Frame和Time列，只保留坐标数据
                numeric_values = [float(v) for v in values[2:] if v.strip()]
                if len(numeric_values) > 0:
                    data_rows.append(numeric_values)
            except ValueError:
                continue
        
        if not data_rows:
            raise ValueError(f"No valid data rows found in {filepath}")
        
        # 转换为numpy数组
        max_cols = max(len(row) for row in data_rows)
        data = np.zeros((len(data_rows), max_cols), dtype=np.float32)
        
        for i, row in enumerate(data_rows):
            data[i, :len(row)] = row
        
        # 准备元数据字典
        metadata = {
            'filepath': filepath,
            'frame_count': len(data_rows),
            'feature_count': max_cols,
            'joint_names': self.joint_names,
            'joint_types': self.joint_types,
            'data_types': self.data_types,
            'file_metadata': self.metadata
        }
        
        return data, metadata
    
    def _parse_metadata(self, line: str):
        """解析元数据行"""
        parts = line.strip().split(',')
        metadata = {}
        for i in range(0, len(parts)-1, 2):
            if i+1 < len(parts):
                key = parts[i].strip()
                value = parts[i+1].strip()
                metadata[key] = value
        self.metadata = metadata
    
    def _parse_types(self, line: str):
        """解析类型行"""
        parts = line.strip().split(',')
        self.joint_types = [p.strip() for p in parts[1:] if p.strip()]  # 跳过第一个空列
    
    def _parse_names(self, line: str):
        """解析关节名称行"""
        parts = line.strip().split(',')
        self.joint_names = [p.strip() for p in parts[1:] if p.strip()]  # 跳过第一个空列
    
    def _parse_data_types(self, line: str):
        """解析数据类型行"""
        parts = line.strip().split(',')
        self.data_types = [p.strip() for p in parts[2:] if p.strip()]  # 跳过Frame和Time列

# -----------------------------------------------------------
# 骨架数据增强
# -----------------------------------------------------------
class SkeletonAugmentation:
    """骨架序列数据增强"""
    
    def __init__(self, 
                 noise_std: float = 0.01,
                 time_stretch_range: Tuple[float, float] = (0.8, 1.2),
                 spatial_scale_range: Tuple[float, float] = (0.95, 1.05),
                 rotation_angle_deg: float = 10.0,  # 修正参数名
                 crop_ratio: float = 0.9):  # 修正参数名
        """
        Args:
            noise_std: 高斯噪声标准差
            time_stretch_range: 时间拉伸范围
            spatial_scale_range: 空间缩放范围  
            rotation_angle_deg: 旋转角度范围（度）
            crop_ratio: 时序裁剪保留比例
        """
        self.noise_std = noise_std
        self.time_stretch_range = time_stretch_range
        self.spatial_scale_range = spatial_scale_range
        self.rotation_range = np.radians(rotation_angle_deg)  # 转换为弧度
        self.temporal_crop_ratio = crop_ratio
    
    def apply_augmentation(self, data: np.ndarray, augment_prob: float = 0.5) -> np.ndarray:
        """
        应用数据增强
        
        Args:
            data: 输入数据 [T, features]
            augment_prob: 每种增强的应用概率
            
        Returns:
            增强后的数据
        """
        augmented = data.copy()
        
        # 1. 添加高斯噪声
        if random.random() < augment_prob:
            augmented = self._add_gaussian_noise(augmented)
        
        # 2. 时间拉伸
        if random.random() < augment_prob:
            augmented = self._time_stretch(augmented)
        
        # 3. 空间缩放
        if random.random() < augment_prob:
            augmented = self._spatial_scaling(augmented)
        
        # 4. 空间旋转（针对位置坐标）
        if random.random() < augment_prob:
            augmented = self._spatial_rotation(augmented)
        
        # 5. 时序裁剪
        if random.random() < augment_prob:
            augmented = self._temporal_crop(augmented)
        
        return augmented
    
    def _add_gaussian_noise(self, data: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_std, data.shape).astype(np.float32)
        return data + noise
    
    def _time_stretch(self, data: np.ndarray) -> np.ndarray:
        """时间拉伸"""
        stretch_factor = random.uniform(*self.time_stretch_range)
        original_length = data.shape[0]
        new_length = int(original_length * stretch_factor)
        
        # 使用线性插值进行时间拉伸
        old_indices = np.linspace(0, original_length - 1, original_length)
        new_indices = np.linspace(0, original_length - 1, new_length)
        
        stretched_data = np.zeros((new_length, data.shape[1]), dtype=np.float32)
        for i in range(data.shape[1]):
            stretched_data[:, i] = np.interp(new_indices, old_indices, data[:, i])
        
        return stretched_data
    
    def _spatial_scaling(self, data: np.ndarray) -> np.ndarray:
        """空间缩放"""
        scale_factor = random.uniform(*self.spatial_scale_range)
        return data * scale_factor
    
    def _spatial_rotation(self, data: np.ndarray) -> np.ndarray:
        """空间旋转（仅针对位置坐标）"""
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # 简单的2D旋转（假设前两个维度是X,Y坐标）
        # 对于更复杂的3D旋转，需要更完整的旋转矩阵
        rotated = data.copy()
        
        # 假设数据格式是连续的坐标组：[x1,y1,z1,qx1,qy1,qz1,qw1, x2,y2,z2,...]
        # 这里简化处理，只对位置坐标应用旋转
        for i in range(0, data.shape[1], 7):  # 假设每7个值为一组（3位置+4四元数）
            if i+1 < data.shape[1]:  # 确保有X,Y坐标
                x = rotated[:, i]
                y = rotated[:, i+1]
                rotated[:, i] = x * cos_a - y * sin_a
                rotated[:, i+1] = x * sin_a + y * cos_a
        
        return rotated
    
    def _temporal_crop(self, data: np.ndarray) -> np.ndarray:
        """时序裁剪"""
        original_length = data.shape[0]
        crop_length = int(original_length * self.temporal_crop_ratio)
        
        if crop_length >= original_length:
            return data
        
        # 随机选择裁剪起始位置
        start_idx = random.randint(0, original_length - crop_length)
        return data[start_idx:start_idx + crop_length]

# -----------------------------------------------------------
# 改进的标签解析
# -----------------------------------------------------------
_action_pat = re.compile(r"action(\d{3})", re.I)

def extract_action_id(fname: str) -> int:
    """从文件名提取动作ID"""
    m = _action_pat.search(fname)
    if not m:
        return -1
    val = int(m.group(1))
    if 1 <= val <= 8:
        return val - 1
    return -1

# -----------------------------------------------------------
# 改进的数据集类
# -----------------------------------------------------------
class EnhancedSkeletonDataset(torch.utils.data.Dataset):
    """增强的骨架数据集，支持数据增强"""
    
    def __init__(self, file_paths: List[str], 
                 enable_augmentation: bool = True,
                 augmentation_config: Optional[Dict] = None):
        super().__init__()
        
        # 初始化解析器和增强器
        self.parser = SkeletonCSVParser()
        
        if enable_augmentation:
            aug_config = augmentation_config or {}
            self.augmenter = SkeletonAugmentation(**aug_config)
        else:
            self.augmenter = None
        
        # 加载数据
        self.samples = []
        self.labels = []
        self.file_paths = []
        
        for file_path in file_paths:
            try:
                # 提取标签
                label = extract_action_id(os.path.basename(file_path))
                if label == -1:
                    print(f"Warning: Skipping invalid file {file_path}")
                    continue
                
                # 解析数据
                data, metadata = self.parser.parse_csv(file_path)
                
                self.samples.append(data)
                self.labels.append(label)
                self.file_paths.append(file_path)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not self.samples:
            raise RuntimeError("No valid samples found!")
        
        # 计算数据集统计信息
        self.max_length = max(sample.shape[0] for sample in self.samples)
        self.max_features = max(sample.shape[1] for sample in self.samples)
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Max sequence length: {self.max_length}")
        print(f"Max feature dimension: {self.max_features}")
        
        # 对数据进行padding和转换
        self._pad_and_convert()
    
    def _pad_and_convert(self):
        """对数据进行padding并转换为tensor"""
        padded_samples = []
        
        for sample in self.samples:
            # 时序padding
            if sample.shape[0] < self.max_length:
                pad_length = self.max_length - sample.shape[0]
                time_pad = np.zeros((pad_length, sample.shape[1]), dtype=np.float32)
                sample = np.vstack([sample, time_pad])
            
            # 特征维padding
            if sample.shape[1] < self.max_features:
                pad_width = self.max_features - sample.shape[1]
                feat_pad = np.zeros((sample.shape[0], pad_width), dtype=np.float32)
                sample = np.hstack([sample, feat_pad])
            
            padded_samples.append(torch.tensor(sample, dtype=torch.float32))
        
        self.samples = padded_samples
        self.labels = [torch.tensor(label, dtype=torch.long) for label in self.labels]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        
        # 在训练时应用数据增强
        if self.augmenter is not None:
            # 转换为numpy进行增强
            sample_np = sample.numpy()
            augmented_np = self.augmenter.apply_augmentation(sample_np)
            
            # 确保增强后的数据形状正确
            if augmented_np.shape[0] != self.max_length:
                # 重新padding
                if augmented_np.shape[0] < self.max_length:
                    pad_length = self.max_length - augmented_np.shape[0]
                    time_pad = np.zeros((pad_length, augmented_np.shape[1]), dtype=np.float32)
                    augmented_np = np.vstack([augmented_np, time_pad])
                else:
                    # 如果超长则截断
                    augmented_np = augmented_np[:self.max_length]
            
            sample = torch.tensor(augmented_np, dtype=torch.float32)
        
        return sample, label
    
    @property
    def feature_dim(self):
        return self.max_features
    
    @feature_dim.setter
    def feature_dim(self, value):
        """设置特征维度"""
        self.max_features = value
    
    @property
    def max_len(self):
        """向后兼容性：max_len别名"""
        return self.max_length

# -----------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------
def create_enhanced_dataset(file_paths: List[str], 
                          is_training: bool = True,
                          augmentation_config: Optional[Dict] = None) -> EnhancedSkeletonDataset:
    """创建增强的骨架数据集"""
    return EnhancedSkeletonDataset(
        file_paths=file_paths,
        enable_augmentation=is_training,  # 只在训练时启用增强
        augmentation_config=augmentation_config
    )

def test_parser(csv_file: str):
    """测试解析器功能"""
    parser = SkeletonCSVParser()
    try:
        data = parser.parse_csv(csv_file)
        print(f"Parsed data shape: {data.shape}")
        print(f"Metadata: {parser.metadata}")
        print(f"First few joint names: {parser.joint_names[:5]}")
        print(f"Data types: {set(parser.data_types)}")
        return data
    except Exception as e:
        print(f"Error parsing {csv_file}: {e}")
        return None

if __name__ == "__main__":
    # 测试解析器
    csv_file = "/Users/yanbojoe/Desktop/AILab/lab4/skeleton/subject000-action001-move_003.csv"
    if os.path.exists(csv_file):
        print("Testing CSV parser...")
        data = test_parser(csv_file)
        if data is not None:
            print(f"Sample data (first 3 rows, first 10 cols):")
            print(data[:3, :10])
