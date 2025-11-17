import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from types import SimpleNamespace

from utils.config import args
from utils.dataset import *
from utils.unitraj import *
from utils.unitraj_diffusion import UniTrajDiffusion
from utils.logger import Logger, log_info
from pathlib import Path
import shutil
from utils.knowledge_base import TrajectoryKnowledgeBase

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"


def main(config, logger):

    # Create the model
    model = UniTrajDiffusion(
        trajectory_length=200,
        patch_size=1,
        embedding_dim=128,
        encoder_layers=8,
        encoder_heads=4,
        mask_ratio=0.5,
        T=1000,   # 训练步数；推理可用 DDIM 20~50 步采样（后续加）
    )

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    file_path = "/vol/zc/UniTraj/data/worldtrace_train.pkl"
    normalize_transform = Normalize()
    dataset = TrajectoryDataset(
        data_path=file_path, max_len=200, transform=normalize_transform
    )
    dataloader = DataLoader(
        dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=32, pin_memory=True)

    val_file_path = "/vol/zc/UniTraj/data/worldtrace_test.pkl"
    dataset_val = TrajectoryDataset(
        data_path= val_file_path,
        max_len=200,
        transform=normalize_transform,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=16,
    )

    # ---------- RAG: 加载知识库 ---------- 
    if hasattr(config, "rag") and getattr(config.rag, "enable", False):
        kb_path = getattr(config.rag, "kb_path", "data/kb_means.pt")
        topk = getattr(config.rag, "topk", 3)
        temperature = getattr(config.rag, "temperature", 0.07)
        inject_prior_in_train = getattr(config.rag, "inject_prior_in_train", False)
        inject_prior_in_sample = getattr(config.rag, "inject_prior_in_sample", True)
        
        # 检查知识库文件是否存在
        if not os.path.exists(kb_path):
            logger.error(f"[RAG] 知识库文件不存在: {kb_path}")
            logger.error("[RAG] 请先运行 build_knowledge_base.py 构建知识库")
            logger.error("[RAG] 命令示例: python build_knowledge_base.py")
            raise FileNotFoundError(f"知识库文件不存在: {kb_path}")
        
        try:
            logger.info(f"[RAG] 加载知识库: {kb_path}")
            kb = TrajectoryKnowledgeBase.load(kb_path)
            
            # 注入到模型（DataParallel 兼容）
            if isinstance(model, torch.nn.DataParallel):
                model.module.set_knowledge_base(kb, topk=topk, temperature=temperature,
                                                inject_prior_in_train=inject_prior_in_train,
                                                inject_prior_in_sample=inject_prior_in_sample)
            else:
                model.set_knowledge_base(kb, topk=topk, temperature=temperature,
                                         inject_prior_in_train=inject_prior_in_train,
                                         inject_prior_in_sample=inject_prior_in_sample)
            logger.info("[RAG] 知识库加载成功")
            logger.info(f"[RAG] 知识库包含 {kb.means.shape[0]} 个轨迹原型")
            
        except Exception as e:
            logger.error(f"[RAG] 知识库加载失败: {e}")
            raise RuntimeError(f"知识库加载失败: {e}")

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)  # Optimizer
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience = 20
    trigger_times = 0
    for epoch in range(0, config.training.n_epochs + 1):
        model.train()
        train_losses = []  # Store losses 
        logger.info("<----- Epoch {} Training ---->".format(epoch))
        for batch_idx, batch in enumerate(dataloader):
            traj, atten_mask = batch["trajectory"], batch["attention_mask"]
            interval, indices = batch["intervals"], batch["indices"]

            # send to device (注意：indices 可能是 LongTensor 或 numpy，需要确认)
            interval = interval.to(device)
            traj = traj.to(device)
            atten_mask = atten_mask.to(device)
            # 如果 indices 是 tensor，放到 device；如果是 numpy 则保持原样（encoder 可能接受 numpy）
            if isinstance(indices, torch.Tensor):
                indices = indices.to(device)

            # atten_mask: [B, L] -> [B,1,L] -> 扩展到与 traj 同 shape [B, C, L]
            atten_mask = atten_mask.unsqueeze(1).expand_as(traj)   # traj [B, C, L]

            # 前向返回噪声预测 eps_hat、真实噪声 eps、以及 mask（被掩码位置），形状假定为 [B,2,L]
            eps_hat, eps, mask = model(traj, interval, indices)   # [B,2,L], [B,2,L], [B,2,L]

            # 若 model 返回的 mask 是 [B,1,L]，请扩展到通道维：
            if mask.dim() == 3 and mask.shape[1] == 1 and traj.shape[1] != 1:
                mask = mask.expand(-1, traj.shape[1], -1)  # [B,2,L]

            # 计算在 mask & atten_mask 上的 MSE
            denom = (mask * atten_mask).sum().clamp_min(1.0)   # 防止除 0
            loss = ((eps_hat - eps) ** 2 * mask * atten_mask).sum() / denom

            # 优化步（保持你当前的风格）
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_losses.append(loss.item())

            # 注意：下面这行通常是用于调试/快跑，训练时请删除或注释
            # break

        avg_train_loss = np.mean(train_losses) if len(train_losses) > 0 else 0.0
        logger.info(f"Epoch {epoch} Training Loss: {avg_train_loss:.5f}")

        # ========== 验证 ==========
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_val):
                traj, atten_mask = batch["trajectory"], batch["attention_mask"]
                interval = batch["intervals"]
                indices = batch["indices"]

                interval = interval.to(device)
                traj = traj.to(device)
                atten_mask = atten_mask.to(device)
                if isinstance(indices, torch.Tensor):
                    indices = indices.to(device)

                atten_mask = atten_mask.unsqueeze(1).expand_as(traj)  

                # 与训练一致，使用噪声回归
                eps_hat, eps, mask = model(traj, interval, indices)

                if mask.dim() == 3 and mask.shape[1] == 1 and traj.shape[1] != 1:
                    mask = mask.expand(-1, traj.shape[1], -1)  # [B,2,L]

                denom = (mask * atten_mask).sum().clamp_min(1.0)
                val_loss = ((eps_hat - eps) ** 2 * mask * atten_mask).sum() / denom

                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses) if len(val_losses) > 0 else 0.0
        logger.info(f"Epoch {epoch} Validation Loss: {avg_val_loss:.5f}")

        scheduler.step(avg_val_loss)
    
        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            # save best model
            m_path = model_save / f"best_model_epoch_{epoch}.pt"
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), m_path)
            else:
                torch.save(model.state_dict(), m_path)
            logger.info(f"Validation loss decreased,\nsaving model to {m_path}")
            
        else:
            trigger_times += 1
            logger.info(f"Validation loss did not decrease for {trigger_times} epochs")
            if trigger_times >= patience:
                m_path = model_save / f"Final_Model_{epoch}.pt"
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), m_path)
                else:
                    torch.save(model.state_dict(), m_path)
                logger.info("Early stopping triggered")
                break

    logger.info("<----Training Done---->")


def setup_experiment_directories(config, Exp_name="UniTraj"):
    root_dir = Path(__file__).resolve().parent
    result_name = f"{config.data.dataset}_bs={config.training.batch_size}"
    exp_dir = root_dir / Exp_name / result_name
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    exp_time_dir = exp_dir / timestamp
    files_save = exp_time_dir / "Files"
    result_save = exp_time_dir / "Results"
    model_save = exp_time_dir / "models"

    # Creating directories
    for directory in [files_save, result_save, model_save]:
        directory.mkdir(parents=True, exist_ok=True)

    # Copying files
    for filename in os.listdir(root_dir / "utils"):
        if filename.endswith(".py"):
            shutil.copy(root_dir / "utils" / filename, files_save)
    # Copying the current file itself
    this_file = Path(__file__)
    shutil.copy(this_file, files_save)

    print("All files saved path ---->>", exp_time_dir)
    logger = Logger(
        __name__, log_path=exp_dir / (timestamp + "/out.log"), colorize=True
    )
    return logger, files_save, result_save, model_save





if __name__ == "__main__":
    # Load configuration
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)

    logger, files_save, result_save, model_save = setup_experiment_directories(
        config, Exp_name="UniTraj"
    )

    log_info(config, logger)
    main(config, logger)
