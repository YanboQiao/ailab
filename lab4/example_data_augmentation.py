# lab4/example_data_augmentation.py - 数据增强使用示例

import torch
import numpy as np
import matplotlib.pyplot as plt
from data import (
    AugmentedSkeletonDataset, 
    create_augmented_dataset,
    SkeletonDataAugmenter,
    preprocess_skeleton_data,
    visualize_augmentation_effect
)


def demo_data_augmentation():
    """演示数据增强效果"""
    
    print("=" * 60)
    print("骨骼数据增强演示")
    print("=" * 60)
    
    # 1. 创建增强数据集
    try:
        train_dataset, test_dataset = create_augmented_dataset(
            data_dir="skeleton",
            train_ratio=0.8,
            augment_prob=0.7,  # 70% 概率进行数据增强
            augment_types=['rotation', 'noise', 'time_warp'],
            seed=42
        )
        
        print(f"✅ 训练集大小: {len(train_dataset)}")
        print(f"✅ 测试集大小: {len(test_dataset)}")
        print(f"✅ 特征维度: {train_dataset.feature_dim}")
        print(f"✅ 最大序列长度: {train_dataset.max_len}")
        
    except Exception as e:
        print(f"❌ 创建数据集失败: {e}")
        return
    
    # 2. 测试数据加载
    print("\n" + "-" * 40)
    print("数据加载测试")
    print("-" * 40)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True
    )
    
    # 获取一个批次的数据
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"✅ 批次 {batch_idx + 1}:")
        print(f"   数据形状: {data.shape}")
        print(f"   标签形状: {labels.shape}")
        print(f"   标签值: {labels.tolist()}")
        
        if batch_idx >= 2:  # 只显示前3个批次
            break
    
    # 3. 演示不同增强方法的效果
    print("\n" + "-" * 40)
    print("数据增强效果演示")
    print("-" * 40)
    
    # 获取一个样本进行增强演示
    sample_data, sample_label = train_dataset[0]
    sample_np = sample_data.numpy()
    
    print(f"原始样本形状: {sample_np.shape}")
    print(f"样本标签: {sample_label.item()}")
    
    # 测试各种增强方法
    augmenter = SkeletonDataAugmenter(
        rotation_range=15.0,
        noise_std=0.02,
        time_warp_ratio=0.1
    )
    
    augment_methods = ['rotation', 'noise', 'time_warp', 'scale', 'translation']
    
    for method in augment_methods:
        try:
            augmented = augmenter.augment(sample_np, [method])
            print(f"✅ {method:12s}: {sample_np.shape} -> {augmented.shape}")
            
            # 计算数据变化程度
            if augmented.shape == sample_np.shape:
                diff = np.mean(np.abs(augmented - sample_np))
                print(f"   平均变化幅度: {diff:.6f}")
            
        except Exception as e:
            print(f"❌ {method:12s}: 错误 - {e}")
    
    # 4. 组合增强演示
    print("\n" + "-" * 40)
    print("组合数据增强演示")
    print("-" * 40)
    
    combinations = [
        ['rotation', 'noise'],
        ['rotation', 'scale'],
        ['noise', 'time_warp'],
        ['rotation', 'noise', 'scale']
    ]
    
    for combo in combinations:
        try:
            augmented = augmenter.augment(sample_np, combo)
            combo_str = '+'.join(combo)
            print(f"✅ {combo_str:20s}: {sample_np.shape} -> {augmented.shape}")
            
        except Exception as e:
            print(f"❌ {combo_str:20s}: 错误 - {e}")


def analyze_augmentation_statistics():
    """分析数据增强的统计特性"""
    
    print("\n" + "=" * 60)
    print("数据增强统计分析")
    print("=" * 60)
    
    try:
        # 创建数据集
        train_dataset, _ = create_augmented_dataset(
            data_dir="skeleton",
            train_ratio=0.9,
            augment_prob=1.0,  # 100% 增强概率用于统计
            augment_types=['rotation', 'noise'],
            seed=42
        )
        
        print(f"数据集大小: {len(train_dataset)}")
        
        # 统计分析
        original_stats = []
        augmented_stats = []
        
        for i in range(min(10, len(train_dataset))):  # 分析前10个样本
            # 获取原始数据（通过临时禁用增强）
            original_prob = train_dataset.augment_prob
            train_dataset.augment_prob = 0.0
            original_data, _ = train_dataset[i]
            
            # 获取增强数据
            train_dataset.augment_prob = 1.0
            augmented_data, _ = train_dataset[i]
            
            # 恢复原始设置
            train_dataset.augment_prob = original_prob
            
            # 计算统计信息
            original_stats.append({
                'mean': float(torch.mean(original_data)),
                'std': float(torch.std(original_data)),
                'min': float(torch.min(original_data)),
                'max': float(torch.max(original_data))
            })
            
            augmented_stats.append({
                'mean': float(torch.mean(augmented_data)),
                'std': float(torch.std(augmented_data)),
                'min': float(torch.min(augmented_data)),
                'max': float(torch.max(augmented_data))
            })
        
        # 输出统计结果
        print("\n原始数据统计:")
        print("样本\t均值\t\t标准差\t\t最小值\t\t最大值")
        for i, stats in enumerate(original_stats):
            print(f"{i}\t{stats['mean']:.4f}\t\t{stats['std']:.4f}\t\t{stats['min']:.4f}\t\t{stats['max']:.4f}")
        
        print("\n增强数据统计:")
        print("样本\t均值\t\t标准差\t\t最小值\t\t最大值")
        for i, stats in enumerate(augmented_stats):
            print(f"{i}\t{stats['mean']:.4f}\t\t{stats['std']:.4f}\t\t{stats['min']:.4f}\t\t{stats['max']:.4f}")
        
        # 计算变化幅度
        print("\n数据增强影响分析:")
        for i in range(len(original_stats)):
            mean_change = abs(augmented_stats[i]['mean'] - original_stats[i]['mean'])
            std_change = abs(augmented_stats[i]['std'] - original_stats[i]['std'])
            print(f"样本 {i}: 均值变化 {mean_change:.6f}, 标准差变化 {std_change:.6f}")
            
    except Exception as e:
        print(f"❌ 统计分析失败: {e}")


def demonstrate_preprocessing_pipeline():
    """演示预处理流水线"""
    
    print("\n" + "=" * 60)
    print("数据预处理流水线演示")
    print("=" * 60)
    
    # 创建模拟数据
    print("创建模拟骨骼数据...")
    
    # 模拟原始CSV数据（包含一些异常值和零值）
    raw_data = np.random.randn(200, 500) * 100  # 200帧，500个特征
    
    # 添加一些异常值
    raw_data[10:15, 50:55] = 0  # 零值帧
    raw_data[20, :] = 10000     # 异常值帧
    raw_data[50:55, 100] = np.nan  # 缺失值
    
    print(f"原始数据形状: {raw_data.shape}")
    print(f"零值数量: {np.sum(raw_data == 0)}")
    print(f"异常值数量: {np.sum(np.abs(raw_data) > 1000)}")
    print(f"缺失值数量: {np.sum(np.isnan(raw_data))}")
    
    # 应用预处理流水线
    print("\n应用预处理流水线...")
    
    processed_data = preprocess_skeleton_data(
        raw_data,
        extract_keys=False,  # 不提取关键关节（因为是模拟数据）
        normalize=True,
        temporal_sample=True,
        sampling_interval=4
    )
    
    print(f"处理后数据形状: {processed_data.shape}")
    print(f"零值数量: {np.sum(processed_data == 0)}")
    print(f"异常值数量: {np.sum(np.abs(processed_data) > 1000)}")
    print(f"缺失值数量: {np.sum(np.isnan(processed_data))}")
    
    # 数据质量检查
    print(f"\n数据质量检查:")
    print(f"数据范围: [{np.min(processed_data):.2f}, {np.max(processed_data):.2f}]")
    print(f"数据均值: {np.mean(processed_data):.4f}")
    print(f"数据标准差: {np.std(processed_data):.4f}")


if __name__ == "__main__":
    # 运行所有演示
    demo_data_augmentation()
    analyze_augmentation_statistics()
    demonstrate_preprocessing_pipeline()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
