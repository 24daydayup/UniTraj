# utils/unitraj_diffusion.py
import torch
import torch.nn as nn
from utils.unitraj import Encoder          # 直接复用你的 Encoder
from utils.denoiser import TrajDenoiser    # 去噪器

class UniTrajDiffusion(nn.Module):
    """
    最小改动版：Encoder + 条件扩散 Denoiser
    - 输入：trajectory [B, 2, L]（已是相对+Normalize），intervals [B, L]，mask_indices [B, M]
    - 输出：eps_hat、eps、mask_2ch（与原 loss 广播逻辑兼容）
    """
    def __init__(self,
                 trajectory_length=200,
                 patch_size=1,
                 embedding_dim=128,
                 encoder_layers=8,
                 encoder_heads=4,
                 mask_ratio=0.5,
                 T=1000):
        super().__init__()
        self.T = T
        # 线性 beta 调度（训练单步）
        betas = torch.linspace(1e-4, 2e-2, T)
        alphas = 1. - betas
        self.register_buffer('alphabar', torch.cumprod(alphas, dim=0))

        # 复用原 UniTraj 的编码器 & 时间间隔嵌入
        self.encoder = Encoder(trajectory_length, patch_size,
                               embedding_dim, encoder_layers, encoder_heads,
                               mask_ratio)
        self.interval_embedding = nn.Linear(1, embedding_dim)

        # 条件去噪器（坐标通道=2，条件通道= x_obs(2) + mask(1) + Δt(1)）
        self.denoiser = TrajDenoiser(in_ch=2, cond_ch=4, hid=embedding_dim, enc_dim=embedding_dim)

    def _noise(self, x0, t):
        """
        x0: [B, 2, L]  ；t: [B] ；返回 x_t, eps
        """
        a = self.alphabar[t].view(-1, 1, 1)  # [B,1,1]
        eps = torch.randn_like(x0)
        x_t = torch.sqrt(a) * x0 + torch.sqrt(1. - a) * eps
        return x_t, eps

    @staticmethod
    def _indices_to_mask(indices, B, L, device):
        """
        indices: [B, M] -> mask [B, 1, L]，掩码位置=1
        """
        mask = torch.zeros(B, L, device=device)
        mask = mask.scatter(1, indices.long(), 1.0)
        return mask.unsqueeze(1)  # [B,1,L]

    def forward(self, trajectory, intervals=None, mask_indices=None):
        """
        训练前向：返回 (eps_hat, eps, mask_2ch)
        """
        B, _, L = trajectory.shape
        device = trajectory.device

        # 1) 时间间隔嵌入
        if intervals is not None:
            intervals = intervals.unsqueeze(-1)                      # [B,L,1]
            interval_embeddings = self.interval_embedding(intervals) # [B,L,C]
        else:
            interval_embeddings = self.interval_embedding(
                torch.zeros(B, L, 1, device=device)
            )

        # 2) Encoder（用可见点编码形成全局上下文）
        mi_np = mask_indices.cpu().numpy() if mask_indices is not None else None
        features, _ = self.encoder(trajectory, interval_embeddings, mi_np)  # [T',B,C]
        enc_feat = features[0]  # 取 CLS token 作为全局上下文 [B,C]

        # 3) 掩码 & 已知点
        mask = self._indices_to_mask(mask_indices.to(device), B, L, device)    # [B,1,L]
        x_obs = trajectory * (1. - mask)                                       # [B,2,L]
        delta_t = intervals.transpose(1, 2) if intervals is not None else torch.zeros(B,1,L,device=device)

        # 4) 采样步与加噪
        t = torch.randint(0, self.T, (B,), device=device)
        x_t, eps = self._noise(trajectory, t)  # 目标噪声

        # 5) 去噪预测（仅在掩码位置监督）
        eps_hat = self.denoiser(x_t, t.float(), x_obs, mask, delta_t, enc_feat=enc_feat)

        # 为了与 main.py 现有广播逻辑兼容，返回 2 通道 mask
        mask_2ch = mask.expand(-1, 2, -1)  # [B,2,L]
        return eps_hat, eps, mask_2ch
