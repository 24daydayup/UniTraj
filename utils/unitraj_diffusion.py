# utils/unitraj_diffusion.py
import torch
import torch.nn as nn
from utils.unitraj import Encoder          # 直接复用你的 Encoder
from utils.denoiser import TrajDenoiser    # 去噪器
from utils.knowledge_base import TrajectoryKnowledgeBase  # 新增：知识库

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
        self.trajectory_length = trajectory_length
        self.patch_size = patch_size
        self.encoder_dim = embedding_dim
        # RAG 相关默认参数
        self.kb: TrajectoryKnowledgeBase | None = None
        self.rag_topk = 3
        self.rag_temperature = 0.07
        self.inject_prior_in_train = False
        self.inject_prior_in_sample = True
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

    # ---------- RAG 设置 ----------
    def set_knowledge_base(
        self,
        kb: TrajectoryKnowledgeBase,
        topk: int = 3,
        temperature: float = 0.07,
        inject_prior_in_train: bool = False,
        inject_prior_in_sample: bool = True,
    ):
        """外部注入知识库与参数。"""
        self.kb = kb
        self.rag_topk = topk
        self.rag_temperature = temperature
        self.inject_prior_in_train = inject_prior_in_train
        self.inject_prior_in_sample = inject_prior_in_sample

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

        # === RAG：全局上下文增强（训练阶段默认不注入轨迹先验） ===
        if self.kb is not None:
            rag_feat, rag_traj = self.kb.retrieve(
                x_obs, mask, intervals, self.encoder, self.interval_embedding,
                topk=self.rag_topk, temperature=self.rag_temperature
            )  # rag_feat:[B,C], rag_traj:[B,2,L]
            enc_feat = enc_feat + rag_feat
            if self.inject_prior_in_train:
                x_obs = x_obs * (1. - mask) + rag_traj * mask

        # 4) 采样步与加噪
        t = torch.randint(0, self.T, (B,), device=device)
        x_t, eps = self._noise(trajectory, t)  # 目标噪声

        # 5) 去噪预测（仅在掩码位置监督）
        eps_hat = self.denoiser(x_t, t.float(), x_obs, mask, delta_t, enc_feat=enc_feat)

        # 为了与 main.py 现有广播逻辑兼容，返回 2 通道 mask
        mask_2ch = mask.expand(-1, 2, -1)  # [B,2,L]
        return eps_hat, eps, mask_2ch

    def forward_diffusion(self, x_t, t, x_obs, mask, delta_t, enc_feat):
        """单步去噪预测，保持时间步为 float 便于 Sinusoidal 嵌入。"""
        if t.dtype != torch.float32:
            t = t.float()
        return self.denoiser(x_t, t, x_obs, mask, delta_t, enc_feat=enc_feat)

    def sample(self, batch_size=1, n_steps=50, device=None):
        """DDIM 采样，返回生成的轨迹序列。"""
        device = torch.device(device or next(self.parameters()).device)
        self.eval()
        with torch.no_grad():
            L = self.trajectory_length
            x_t = torch.randn((batch_size, 2, L), device=device)
            mask = torch.ones(batch_size, 1, L, device=device)
            x_obs = torch.zeros_like(x_t)
            delta_t = torch.zeros(batch_size, 1, L, device=device)
            enc_feat = torch.zeros(batch_size, self.encoder_dim, device=device)

            # 可选：无条件采样也可用 RAG 全局上下文（无观测 → 查询为全零，会退化为均匀注意力）
            if self.kb is not None:
                rag_feat, rag_traj = self.kb.retrieve(
                    x_obs, mask, None, self.encoder, self.interval_embedding,
                    topk=self.rag_topk, temperature=self.rag_temperature
                )
                enc_feat = enc_feat + rag_feat
                if self.inject_prior_in_sample:
                    x_obs = rag_traj.clone() * mask  # 无条件场景下，用先验填充缺失位
            else:
                rag_feat = None

            times = torch.linspace(self.T - 1, 0, steps=n_steps + 1, device=device, dtype=torch.long)

            for i in range(n_steps):
                t_now = times[i]
                t_next = times[i + 1]

                t_batch = torch.full((batch_size,), t_now, device=device, dtype=torch.long)
                eps_pred = self.forward_diffusion(x_t, t_batch, x_obs, mask, delta_t, enc_feat)

                alphabar_now = self.alphabar[t_now].view(1, 1, 1)
                alphabar_next = self.alphabar[t_next].view(1, 1, 1)

                x0_pred = (x_t - torch.sqrt(1.0 - alphabar_now) * eps_pred) / torch.sqrt(alphabar_now)
                x_t = torch.sqrt(alphabar_next) * x0_pred + torch.sqrt(1.0 - alphabar_next) * eps_pred

        return x_t
