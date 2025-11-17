import os
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List

from .dataset import TrajectoryDataset, Normalize  # 直接复用你的数据预处理


class TrajectoryKnowledgeBase:
    """
    轨迹知识库（均值轨迹原型 + 对应的编码向量），用于检索增强。
    - means: [K, 2, L] 归一化后的均值轨迹（与训练/推理输入同空间，同长度）
    - encodings: [K, C] 使用 Encoder 的 CLS 编码得到的检索键
    """
    def __init__(self, means: torch.Tensor, traj_length: int, emb_dim: int):
        assert means.dim() == 3 and means.size(1) == 2, "means should be [K, 2, L]"
        self.means = means.float().contiguous()            # [K, 2, L]
        self.traj_length = traj_length
        self.emb_dim = emb_dim
        self._encodings: Optional[torch.Tensor] = None     # [K, C]
        self._enc_device: Optional[torch.device] = None

    # ---------- 构建 ----------
    @staticmethod
    def build_from_dataset(
        data_path: str,
        max_len: int = 200,
        k: int = 256,
        sample_size: int = 2500000,
        num_workers: int = 32,
        seed: int = 42,
    ) -> "TrajectoryKnowledgeBase":
        """
        从数据集构建 K 个均值轨迹（KMeans 简化实现）。
        说明：
          - 直接使用 TrajectoryDataset 的输出（[2, L] + padding），简单有效；
          - 计算均值时对 padding 使用 attention_mask 做掩码平均。
        """
        rng = np.random.RandomState(seed)
        transform = Normalize()
        ds = TrajectoryDataset(data_path=data_path, max_len=max_len, transform=transform, mask_ratio=0.5)
        # 采样（避免超大规模内存）
        indices = np.arange(len(ds))
        if len(indices) > sample_size:
            indices = rng.choice(indices, size=sample_size, replace=False)
        # 收集样本
        X, M = [], []
        for i in indices:
            item = ds[i]
            X.append(item["trajectory"].numpy())      # [2, L]
            M.append(item["attention_mask"].numpy())  # [L]
        X = np.stack(X, axis=0)                       # [N, 2, L]
        M = np.stack(M, axis=0)                       # [N, L]

        N, _, L = X.shape
        X_flat = X.reshape(N, -1)                     # [N, 2L]

        # ------- KMeans（简化版）-------
        K = min(k, N)
        # kmeans++ 初始化
        centers = _kmeans_plus_plus_init(X_flat, K, rng=rng)
        prev_inertia = None
        for _ in range(50):
            # 分配
            dists = _squared_cdist(X_flat, centers)          # [N, K]
            labels = dists.argmin(axis=1)                    # [N]
            # 更新
            new_centers = np.zeros_like(centers)
            for c in range(K):
                idx = (labels == c)
                if idx.any():
                    new_centers[c] = X_flat[idx].mean(axis=0)
                else:
                    new_centers[c] = centers[rng.randint(0, N)]
            # 收敛判断
            inertia = np.mean(np.min(_squared_cdist(X_flat, new_centers), axis=1))
            if prev_inertia is not None and abs(prev_inertia - inertia) < 1e-6:
                centers = new_centers
                break
            centers = new_centers
            prev_inertia = inertia

        # ------- 基于掩码的均值轨迹 -------
        means = np.zeros((K, 2, L), dtype=np.float32)
        for c in range(K):
            idx = (labels == c)
            if not idx.any():
                continue
            Xc = X[idx]                    # [Nc, 2, L]
            Mc = M[idx][:, None, :]        # [Nc, 1, L]
            denom = Mc.sum(axis=0, keepdims=False).clip(min=1.0)  # [1, L] -> [1,L]
            means[c] = (Xc * Mc).sum(axis=0) / denom               # [2, L]

        means = torch.from_numpy(means)  # [K, 2, L]
        return TrajectoryKnowledgeBase(means, traj_length=L, emb_dim=128)

    # ---------- 保存 / 加载 ----------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"means": self.means, "traj_length": self.traj_length, "emb_dim": self.emb_dim}, path)

    @staticmethod
    def load(path: str) -> "TrajectoryKnowledgeBase":
        blob = torch.load(path, map_location="cpu")
        kb = TrajectoryKnowledgeBase(blob["means"], traj_length=int(blob["traj_length"]), emb_dim=int(blob["emb_dim"]))
        return kb

    # ---------- 编码 & 检索 ----------
    def _ensure_encodings(self, encoder, interval_embedding, device: torch.device, batch_size: int = 512):
        """
        用当前 encoder 对库内均值轨迹编码，缓存为 encodings（检索键）。
        """
        if self._encodings is not None and self._enc_device == device:
            return
        K, _, L = self.means.shape
        encs = []
        with torch.no_grad():
            for s in range(0, K, batch_size):
                chunk = self.means[s : s + batch_size].to(device)          # [b, 2, L]，chunk块
                intervals = torch.zeros(chunk.size(0), L, 1, device=device)
                interval_emb = interval_embedding(intervals)                # [b, L, C]
                feats, _ = encoder(chunk, interval_emb, mask_indices=None)  # [T, b, C]
                cls = feats[0]                                             # [b, C]
                encs.append(cls.detach().cpu())
        self._encodings = torch.cat(encs, dim=0)                           # [K, C]
        self._enc_device = device

    def retrieve(
        self,
        x_obs: torch.Tensor,           # [B, 2, L]
        mask: torch.Tensor,            # [B, 1, L]  (1=缺失，0=可见)
        intervals: Optional[torch.Tensor],   # [B, L] 或 None
        encoder,                       # 直接传入 model.encoder
        interval_embedding,            # 直接传入 model.interval_embedding (nn.Linear)
        topk: int = 3,
        temperature: float = 0.07,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (rag_feat, rag_traj)
          - rag_feat: [B, C]  Top‑k CLS 加权和
          - rag_traj: [B, 2, L]  Top‑k 均值轨迹加权和
        """
        device = x_obs.device
        B, _, L = x_obs.shape  #
        self._ensure_encodings(encoder, interval_embedding, device=device)

        # 编码 query（只基于可见点；mask=1 表示缺失，我们把这些索引传给 encoder 以跳过）
        if intervals is not None:
            # 兼容 [B, L] 或 [B, L, 1]
            if intervals.dim() == 2:
                intervals_in = intervals.unsqueeze(-1)     # [B, L, 1]
            else:
                intervals_in = intervals                   # 已是 [B, L, 1]
            intervals_emb = interval_embedding(intervals_in)  # [B, L, C]
        else:
            intervals_emb = interval_embedding(torch.zeros(B, L, 1, device=device))

        # 构造每个样本的 mask_indices
        mask_indices = [torch.where(mask[b, 0] == 1)[0].cpu().numpy() for b in range(B)]
        feats, _ = encoder(x_obs, intervals_emb, mask_indices)
        q = feats[0]                                  # [B, C]

        # 相似度（余弦）
        keys = self._encodings.to(device)             # [K, C]
        qn = torch.nn.functional.normalize(q, dim=-1)
        kn = torch.nn.functional.normalize(keys, dim=-1)
        sim = torch.matmul(qn, kn.t())                # [B, K]
        vals, idx = sim.topk(k=topk, dim=-1)          # [B, topk]
        attn = torch.softmax(vals / temperature, dim=-1)  # [B, topk]

        # 聚合 rag_feat
        gathered_keys = torch.gather(kn.unsqueeze(0).expand(B, -1, -1), 1, idx.unsqueeze(-1).expand(-1, -1, kn.size(-1)))
        rag_feat = (attn.unsqueeze(-1) * gathered_keys).sum(dim=1) * q.norm(dim=-1, keepdim=True)  # [B, C]（尺度对齐）

        # 聚合 rag_traj
        means = self.means.to(device)                 # [K, 2, L]
        gathered_means = torch.gather(
            means.unsqueeze(0).expand(B, -1, -1, -1), 1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, means.size(1), means.size(2))
        )                                             # [B, topk, 2, L]
        rag_traj = (attn.view(B, topk, 1, 1) * gathered_means).sum(dim=1)  # [B, 2, L]
        return rag_feat, rag_traj


# ----------------- 工具函数 -----------------
def _squared_cdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: [N, D], B: [K, D]
    # 返回 [N, K]
    # (a-b)^2 = a^2 + b^2 - 2ab
    a2 = (A * A).sum(axis=1, keepdims=True)
    b2 = (B * B).sum(axis=1, keepdims=True).T
    ab = A @ B.T
    return a2 + b2 - 2.0 * ab

def _kmeans_plus_plus_init(X: np.ndarray, K: int, rng: np.random.RandomState) -> np.ndarray:
    N, D = X.shape
    centers = np.empty((K, D), dtype=X.dtype)
    # 1) 随机挑第一个
    idx = rng.randint(0, N)
    centers[0] = X[idx]
    # 2) 依次用距离^2 做加权采样
    closest_dist_sq = _squared_cdist(X, centers[0:1]).reshape(N)
    # 确保距离平方值为非负
    closest_dist_sq = np.maximum(closest_dist_sq, 0.0)
    for c in range(1, K):
        # 确保概率计算稳定：避免除零和负概率
        dist_sum = closest_dist_sq.sum()
        if dist_sum <= 0:
            # 如果所有距离都是0，使用均匀分布
            probs = np.ones(N) / N
        else:
            probs = closest_dist_sq / dist_sum
        
        # 确保概率是有效的（非负且和为1）
        probs = np.maximum(probs, 0.0)
        probs = probs / probs.sum()
        
        idx = rng.choice(N, p=probs)
        centers[c] = X[idx]
        dist_sq = _squared_cdist(X, centers[c:c+1]).reshape(N)
        dist_sq = np.maximum(dist_sq, 0.0)  # 确保距离平方非负
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
    return centers