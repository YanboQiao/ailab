#!/usr/bin/env python3
"""
测试增强的骨架数据处理模块与现有训练流水线的集成
"""

import os
import torch
from skeleton_utils import EnhancedSkeletonDataset

def test_enhanced_dataset():
    """测试增强数据集的基本功能"""
    print("=" * 60)
    print("测试增强骨架数据集")
    print("=" * 60)
    
    # 找到CSV文件
    data_dir = "skeleton"
    csv_files = [os.path.join(data_dir, f) 
                 for f in os.listdir(data_dir) 
                 if f.lower().endswith(".csv")]
    
    if not csv_files:
        print("❌ 未找到CSV文件")
        return False
        
    print(f"✓ 找到 {len(csv_files)} 个CSV文件")
    
    # 测试无增强的数据集
    print("\n1. 测试基础数据集 (无增强)...")
    try:
        basic_ds = EnhancedSkeletonDataset(csv_files[:2])
        print(f"✓ 基础数据集创建成功")
        print(f"  - 样本数量: {len(basic_ds)}")
        print(f"  - 特征维度: {basic_ds.feature_dim}")
        print(f"  - 最大序列长度: {basic_ds.max_len}")
        
        # 测试获取单个样本
        sample, label = basic_ds[0]
        print(f"  - 第一个样本形状: {sample.shape}")
        print(f"  - 第一个样本标签: {label}")
        
    except Exception as e:
        print(f"❌ 基础数据集测试失败: {e}")
        return False
    
    # 测试带增强的数据集
    print("\n2. 测试增强数据集...")
    try:
        augmentation_config = {
            'noise_std': 0.02,
            'time_stretch_range': [0.8, 1.2],
            'spatial_scale_range': [0.9, 1.1],
            'rotation_angle_deg': 15.0,
            'crop_ratio': 0.1
        }
        aug_ds = EnhancedSkeletonDataset(csv_files[:2], augmentation_config=augmentation_config)
        print(f"✓ 增强数据集创建成功")
        print(f"  - 样本数量: {len(aug_ds)}")
        print(f"  - 特征维度: {aug_ds.feature_dim}")
        
        # 测试多次获取同一样本，验证增强的随机性
        sample1, _ = aug_ds[0]
        sample2, _ = aug_ds[0]
        
        if torch.equal(sample1, sample2):
            print("  ⚠️  警告: 增强可能未生效，两次获取相同样本结果相同")
        else:
            print("  ✓ 数据增强正常工作 (相同索引返回不同结果)")
            
    except Exception as e:
        print(f"❌ 增强数据集测试失败: {e}")
        return False
    
    # 测试DataLoader兼容性
    print("\n3. 测试DataLoader兼容性...")
    try:
        dataloader = torch.utils.data.DataLoader(
            basic_ds, batch_size=4, shuffle=True
        )
        
        batch_x, batch_y = next(iter(dataloader))
        print(f"✓ DataLoader兼容性测试通过")
        print(f"  - 批次输入形状: {batch_x.shape}")
        print(f"  - 批次标签形状: {batch_y.shape}")
        
    except Exception as e:
        print(f"❌ DataLoader兼容性测试失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！增强数据集集成成功")
    print("=" * 60)
    return True

def test_csv_parsing():
    """测试CSV解析的详细信息"""
    print("\n" + "=" * 60)
    print("测试CSV解析详情")
    print("=" * 60)
    
    from skeleton_utils import SkeletonCSVParser
    
    # 找到一个CSV文件进行详细测试
    data_dir = "skeleton"
    csv_files = [os.path.join(data_dir, f) 
                 for f in os.listdir(data_dir) 
                 if f.lower().endswith(".csv")]
    
    if not csv_files:
        print("❌ 未找到CSV文件")
        return False
    
    csv_file = csv_files[0]
    print(f"测试文件: {os.path.basename(csv_file)}")
    
    try:
        parser = SkeletonCSVParser()
        data, metadata = parser.parse_csv(csv_file)
        
        print(f"✓ CSV解析成功")
        print(f"  - 数据形状: {data.shape}")
        print(f"  - 元数据:")
        for key, value in metadata.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"    {key}: [{len(value)} items] {value[:3]}...")
            else:
                print(f"    {key}: {value}")
                
    except Exception as e:
        print(f"❌ CSV解析失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = True
    success &= test_csv_parsing()
    success &= test_enhanced_dataset()
    
    if success:
        print("\n🎉 集成测试完全成功！可以开始使用增强的数据处理流水线。")
    else:
        print("\n💥 集成测试失败，请检查错误信息并修复问题。")
