# utils/denoiser.py
import math
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim*4)
        )

    def forward(self, t):  # t: [B]，标量步数
        half = 64
        device = t.device
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device)/half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 128]
        return self.mlp(emb)  # [B, 512]

class TrajDenoiser(nn.Module):
    """
    条件去噪器：输入 x_t、x_obs、mask、Δt，以及 encoder 的全局上下文 enc_feat（CLS/mean）。
    预测 epsilon（与 DDPM 训练目标匹配）。
    """
    def __init__(self, in_ch=2, cond_ch=2+1+1, hid=128, enc_dim=128):
        super().__init__()
        self.time_embed = TimeEmbedding(dim=128)
        self.time_proj = nn.Linear(512, hid)

        # encoder 全局上下文投影（FiLM 风格可选，这里做加性注入）
        self.enc_proj = nn.Linear(enc_dim, hid)

        # x_t 与条件拼接后进入卷积主干
        self.inp = nn.Conv1d(in_ch + cond_ch, hid, 3, padding=1)
        self.block1 = nn.Sequential(nn.GroupNorm(8, hid), nn.SiLU(),
                                    nn.Conv1d(hid, hid, 3, padding=1))
        self.block2 = nn.Sequential(nn.GroupNorm(8, hid), nn.SiLU(),
                                    nn.Conv1d(hid, hid, 3, padding=1))
        self.out = nn.Conv1d(hid, 2, 3, padding=1)  # 预测 epsilon（2 维坐标）

    def forward(self, x_t, t, x_obs, mask, delta_t, enc_feat=None):
        """
        x_t/x_obs: [B, 2, L]；mask/delta_t: [B, 1, L]；enc_feat: [B, C]
        """
        cond = torch.cat([x_obs, mask, delta_t], dim=1)   # [B, 4, L]
        h = torch.cat([x_t, cond], dim=1)                 # [B, 6, L]
        h = self.inp(h)

        # 时间步 & 全局上下文注入
        ht = self.time_proj(self.time_embed(t))[:, :, None]  # [B, hid, 1]
        h = h + ht
        if enc_feat is not None:
            he = self.enc_proj(enc_feat)[:, :, None]         # [B, hid, 1]
            h = h + he

        h = h + self.block1(h)
        h = h + self.block2(h)
        eps_hat = self.out(h)                                 # [B, 2, L]
        return eps_hat