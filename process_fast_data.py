#!/usr/bin/env python3
"""
快速数据处理脚本：快速处理所有文件对并生成包含实际数据的.pkl文件
"""

import os
import json
import pandas as pd
import numpy as np
import pickle
import glob
from typing import List, Tuple

def get_file_pairs_fast(meta_dir: str, trajectory_dir: str) -> List[Tuple[str, str]]:
    """
    快速获取配对的元数据文件和轨迹数据文件
    """
    # 获取所有元数据JSON文件
    meta_files = glob.glob(os.path.join(meta_dir, "*.json"))
    
    file_pairs = []
    
    for meta_file in meta_files:
        # 从元数据文件名提取ID
        file_id = os.path.basename(meta_file).replace('.json', '')
        
        # 对应的轨迹数据文件
        trajectory_file = os.path.join(trajectory_dir, f"{file_id}.csv")
        
        # 检查轨迹数据文件是否存在
        if os.path.exists(trajectory_file):
            file_pairs.append((meta_file, trajectory_file))
    
    return file_pairs

def process_trajectory_pair_fast(meta_file: str, trajectory_file: str) -> dict:
    """
    快速处理一对元数据和轨迹数据文件
    """
    try:
        # 加载元数据
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        # 加载轨迹数据
        trajectory_df = pd.read_csv(trajectory_file)
        
        # 转换时间列为datetime类型
        trajectory_df['time'] = pd.to_datetime(trajectory_df['time'])
        
        # 提取轨迹点数据
        trajectory_points = trajectory_df[['latitude', 'longitude']].values
        
        # 提取时间数据
        time_data = trajectory_df['time']
        
        # 创建结果字典
        result = {
            'time': time_data,
            'trajectory': trajectory_points,
            'meta_file': meta_file,
            'trajectory_file': trajectory_file,
            'file_id': os.path.basename(meta_file).replace('.json', '')
        }
        
        return result
        
    except Exception as e:
        print(f"处理文件 {meta_file} 时出错: {e}")
        return None

def create_fast_dataset(meta_dir: str, trajectory_dir: str, output_file: str, test_ratio: float = 0.1, random_seed: int = 42) -> None:
    """
    创建快速数据集
    """
    print("开始快速处理完整数据...")
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 获取文件对
    file_pairs = get_file_pairs_fast(meta_dir, trajectory_dir)
    print(f"找到 {len(file_pairs)} 对数据文件")
    
    # 处理所有文件对
    processed_data = []
    
    for i, (meta_file, trajectory_file) in enumerate(file_pairs):
        if (i + 1) % 1000 == 0:
            print(f"处理第 {i+1}/{len(file_pairs)} 对文件")
        
        # 处理数据对
        trajectory_data = process_trajectory_pair_fast(meta_file, trajectory_file)
        if trajectory_data is not None:
            processed_data.append(trajectory_data)
    
    print(f"成功处理 {len(processed_data)} 条轨迹数据（100%完整数据）")
    
    # 创建DataFrame
    df = pd.DataFrame(processed_data)
    
    # 随机打乱数据
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # 计算测试集大小
    test_size = int(len(df_shuffled) * test_ratio)
    train_size = len(df_shuffled) - test_size
    
    # 分割数据集
    test_df = df_shuffled.iloc[:test_size]
    train_df = df_shuffled.iloc[test_size:]
    
    # 保存训练集
    train_output_file = output_file.replace('.pkl', '_train.pkl')
    with open(train_output_file, 'wb') as f:
        pickle.dump(train_df, f)
    
    # 保存测试集
    test_output_file = output_file.replace('.pkl', '_test.pkl')
    with open(test_output_file, 'wb') as f:
        pickle.dump(test_df, f)
    
    print(f"训练集已保存到: {train_output_file}")
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集已保存到: {test_output_file}")
    print(f"测试集形状: {test_df.shape}")
    print(f"分割比例: 训练集 {train_size} 条 ({train_size/len(df_shuffled)*100:.1f}%), "
          f"测试集 {test_size} 条 ({test_size/len(df_shuffled)*100:.1f}%)")
    print(f"列名: {train_df.columns.tolist()}")
    print("✓ 100%完整数据处理完成！")

def main():
    """主函数"""
    # 设置路径
    meta_dir = "/vol/zc/UniTraj/modelscope/Meta/data/yuanshao/OpenTrace/Meta"
    trajectory_dir = "/vol/zc/UniTraj/modelscope/Trajectory/data/yuanshao/OpenTrace/Trajectory"
    output_file = "data/worldtrace_full.pkl"
    
    print(f"元数据目录: {meta_dir}")
    print(f"轨迹数据目录: {trajectory_dir}")
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 处理完整数据
    create_fast_dataset(meta_dir, trajectory_dir, output_file, test_ratio=0.1, random_seed=42)

if __name__ == "__main__":
    main()