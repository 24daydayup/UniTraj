#!/usr/bin/env python3
"""
独立的知识库构建脚本
在训练前先构建知识库，避免训练过程中重复构建
"""

import os
import sys
import argparse
from pathlib import Path
from types import SimpleNamespace

# 添加项目路径到系统路径
sys.path.append(str(Path(__file__).parent))

from utils.knowledge_base import TrajectoryKnowledgeBase
from utils.logger import Logger
from utils.config import args


def build_knowledge_base(config, data_path, kb_path, logger=None):
    """
    构建知识库
    
    Args:
        config: 配置对象
        data_path: 训练数据路径
        kb_path: 知识库保存路径
        logger: 日志记录器
    """
    if logger is None:
        logger = Logger(__name__)
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 检查知识库是否已存在
    if os.path.exists(kb_path):
        logger.info(f"知识库已存在: {kb_path}")
        logger.info("跳过知识库构建...")
        return True
    
    logger.info("开始构建知识库...")
    logger.info(f"数据文件: {data_path}")
    logger.info(f"知识库保存路径: {kb_path}")
    
    try:
        # 构建知识库
        kb = TrajectoryKnowledgeBase.build_from_dataset(
            data_path=data_path,
            max_len=config.data.traj_length,
            k=getattr(config.rag, "k", 256),  # 知识库存的均值轨迹个数
            sample_size=2500000,   # 采样样本的上限
            num_workers=32,  # 减少worker数量以避免内存问题
            seed=42
        )
        
        # 保存知识库
        kb.save(kb_path)
        logger.info(f"知识库构建完成并保存到: {kb_path}")
        
        # 输出知识库统计信息
        logger.info(f"知识库包含 {kb.means.shape[0]} 个轨迹原型")
        logger.info(f"轨迹长度: {kb.traj_length}")
        logger.info(f"嵌入维度: {kb.emb_dim}")
        
        return True
        
    except Exception as e:
        logger.error(f"知识库构建失败: {e}")
        # 如果构建失败，删除可能创建的不完整文件
        if os.path.exists(kb_path):
            os.remove(kb_path)
            logger.info(f"已删除不完整的知识库文件: {kb_path}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='构建轨迹知识库')
    parser.add_argument('--data_path', type=str, default='data/worldtrace_train.pkl',
                       help='训练数据文件路径')
    parser.add_argument('--kb_path', type=str, default='data/kb_means.pt',
                       help='知识库保存路径')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（可选）')
    
    args_cmd = parser.parse_args()
    
    # 加载配置
    if args_cmd.config:
        # 从文件加载配置（如果需要）
        # 这里简化处理，使用默认配置
        pass
    
    # 使用项目默认配置
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)
    
    # 创建日志记录器
    logger = Logger(__name__, log_path="knowledge_base_build.log", colorize=True)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args_cmd.kb_path), exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("知识库构建脚本启动")
    logger.info("=" * 50)   #50次*
    
    # 构建知识库
    success = build_knowledge_base(
        config=config,
        data_path=args_cmd.data_path,
        kb_path=args_cmd.kb_path,
        logger=logger
    )
    
    if success:
        logger.info("知识库构建完成！")
        logger.info("现在可以运行训练脚本，训练时会自动加载已构建的知识库")
    else:
        logger.error("知识库构建失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()