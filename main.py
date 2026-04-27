import os
import math
import glob
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz_embedding(path: str, key: str) -> torch.Tensor:
    """
    Load one protein embedding from a packed .npz file.
    key example: "prot_000987__tok"
    Returns: FloatTensor [L, 1280]
    """
    with np.load(path, allow_pickle=True) as z:
        if key not in z:
            raise KeyError(f"Key '{key}' not found in {path}. Available keys example: {list(z.keys())[:5]}")
        arr = z[key]

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Embedding must be 2D [L,D], got shape={arr.shape} from {path}:{key}")
    if arr.shape[1] != 1280:
        raise ValueError(f"Expected dim=1280, got {arr.shape[1]} from {path}:{key}")
    return torch.from_numpy(arr).float()


# Dataset
import os, glob, re, random
from torch.utils.data import Dataset

class ProteinEmbeddingDataset(Dataset):
    """
    Each sample = one protein embedding stored under a key inside a packed .npz.
    Directory layout:
      Train_neg/  (label 0)  contains prot_emb_000.npz ...
      Train_pos/  (label 1)
      Test_neg/
      Test_pos/
    """
    def __init__(self, base_dir: str, split: str):
        super().__init__()
        assert split in ["Train", "Test"]

        neg_dir = os.path.join(base_dir, f"{split}_neg")
        pos_dir = os.path.join(base_dir, f"{split}_pos")

        neg_npzs = sorted(glob.glob(os.path.join(neg_dir, "prot_emb_*.npz")))
        pos_npzs = sorted(glob.glob(os.path.join(pos_dir, "prot_emb_*.npz")))

        if len(neg_npzs) == 0 or len(pos_npzs) == 0:
            raise ValueError(f"No npz files found under {neg_dir} or {pos_dir}")

        key_pat = re.compile(r"^prot_\d+__tok$")

        # samples: List[Tuple[npz_path, key, label]]
        self.samples = []

        def add_npz(npz_path: str, label: int):
            with np.load(npz_path, allow_pickle=True) as z:
                keys = [k for k in z.keys() if key_pat.match(k)]
            if len(keys) == 0:
                raise ValueError(f"No prot_*__tok keys found in {npz_path}")
            for k in keys:
                self.samples.append((npz_path, k, label))

        for p in neg_npzs:
            add_npz(p, 0)
        for p in pos_npzs:
            add_npz(p, 1)

        random.shuffle(self.samples)

        self._cache_path = None
        self._cache_npz = None

    def __len__(self):
        return len(self.samples)

    def _get_npz(self, path: str):
        if self._cache_path != path:
            if self._cache_npz is not None:
                try:
                    self._cache_npz.close()
                except Exception:
                    pass
            self._cache_npz = np.load(path, allow_pickle=True)
            self._cache_path = path
        return self._cache_npz

    def __getitem__(self, idx: int):
        npz_path, key, label = self.samples[idx]
        z = self._get_npz(npz_path)
        arr = np.asarray(z[key])

        if arr.ndim != 2 or arr.shape[1] != 1280:
            raise ValueError(f"Bad embedding shape {arr.shape} in {npz_path}:{key}")

        emb = torch.from_numpy(arr).float()
        return {"emb": emb, "label": int(label), "path": npz_path, "key": key}


def pad_batch(seqs: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seqs: list of [L_i, D]
    returns:
      x:    [B, Lmax, D]
      mask: [B, Lmax] (1 valid, 0 pad)
    """
    B = len(seqs)
    Lmax = max(s.size(0) for s in seqs)
    D = seqs[0].size(1)
    x = seqs[0].new_full((B, Lmax, D), pad_value)
    mask = seqs[0].new_zeros((B, Lmax), dtype=torch.long)
    for b, s in enumerate(seqs):
        L = s.size(0)
        x[b, :L] = s
        mask[b, :L] = 1
    return x, mask


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    embs = [b["emb"] for b in batch]
    x, mask = pad_batch(embs, pad_value=0.0)
    MAX_LEN = 1000
    if x.size(1) > MAX_LEN:
        x = x[:, :MAX_LEN].contiguous()
        mask = mask[:, :MAX_LEN].contiguous()
    
    y = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    paths = [b["path"] for b in batch]
    # print({"x": x, "mask": mask, "y": y, "paths": paths})
    # raise
    return {"x": x, "mask": mask, "y": y, "paths": paths}


def make_global_view(x: torch.Tensor, mask: torch.Tensor,
                     noise_std: float = 0.01, token_dropout: float = 0.05) -> torch.Tensor:
    """
    Mild global augmentation (teacher & student can use)
    """
    x = x + noise_std * torch.randn_like(x)
    if token_dropout > 0:
        drop = (torch.rand_like(mask.float()) < token_dropout) & (mask == 1)
        x = x.masked_fill(drop.unsqueeze(-1), 0.0)
    return x


def random_crop_batch(x: torch.Tensor, mask: torch.Tensor,
                      crop_ratio_min: float, crop_ratio_max: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crop each sequence independently, then repad.
    """
    B, Lmax, D = x.shape
    seqs = []
    for b in range(B):
        L = int(mask[b].sum().item())
        if L <= 2:
            seqs.append(x[b, :L])
            continue
        ratio = random.uniform(crop_ratio_min, crop_ratio_max)
        clen = max(2, int(L * ratio))
        start = random.randint(0, max(0, L - clen))
        seqs.append(x[b, start:start+clen])
    return pad_batch(seqs, pad_value=0.0)


def make_local_view(x: torch.Tensor, mask: torch.Tensor,
                    crop_ratio_min: float = 0.4, crop_ratio_max: float = 0.8,
                    noise_std: float = 0.02, token_dropout: float = 0.10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stronger local augmentation (for student)
    """
    x2, m2 = random_crop_batch(x, mask, crop_ratio_min, crop_ratio_max)
    x2 = x2 + noise_std * torch.randn_like(x2)
    if token_dropout > 0:
        drop = (torch.rand_like(m2.float()) < token_dropout) & (m2 == 1)
        x2 = x2.masked_fill(drop.unsqueeze(-1), 0.0)
    return x2, m2


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(mask == 0, neg_inf)
    return torch.softmax(logits, dim=dim)


# Dynamic kNN graph
@torch.no_grad()
def build_dynamic_knn(x: torch.Tensor, mask: torch.Tensor, k: int = 32, add_local: int = 2,
                      max_tokens_for_knn: int = 2500) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build per-sample neighbor indices from embedding similarity.
    This does NOT require ESM2 attention map.
    """
    device = x.device
    B, L, D = x.shape
    K = k + 2 * add_local

    nbr_idx = torch.zeros((B, L, K), dtype=torch.long, device=device)
    nbr_mask = torch.zeros((B, L, K), dtype=torch.long, device=device)
    rel_pos = torch.zeros((B, L, K), dtype=torch.long, device=device)

    for b in range(B):
        n = int(mask[b].sum().item())
        if n <= 1:
            continue

        n_eff = min(n, max_tokens_for_knn)
        h = x[b, :n_eff].float()
        h = F.normalize(h, dim=-1)

        # compute sim in chunks to reduce peak memory
        kk = min(k, max(1, n_eff - 1))
        topk_idx = torch.zeros((n_eff, kk), dtype=torch.long, device=device)

        chunk = 512
        for i0 in range(0, n_eff, chunk):
            i1 = min(n_eff, i0 + chunk)
            q = h[i0:i1]                           # [c, D]
            sim = q @ h.t()                        # [c, n_eff]
            # exclude self where applicable
            rows = torch.arange(i0, i1, device=device)
            sim[torch.arange(i1 - i0, device=device), rows] = -1e9
            topk_idx[i0:i1] = torch.topk(sim, kk, dim=-1).indices

        # fill neighbors
        for i in range(n):
            if i < n_eff:
                sem = topk_idx[i].tolist()
            else:
                sem = []  

            locals_ = []
            for off in range(1, add_local + 1):
                if i - off >= 0:
                    locals_.append(i - off)
                if i + off < n:
                    locals_.append(i + off)

            nb = sem + locals_
            nb = nb[:K]
            while len(nb) < K:
                nb.append(0)

            nb_t = torch.tensor(nb, device=device, dtype=torch.long)
            nbr_idx[b, i] = nb_t
            valid_count = min(K, len(sem) + len(locals_))
            nbr_mask[b, i, :valid_count] = 1
            rel_pos[b, i] = nb_t - i

    return nbr_idx, nbr_mask, rel_pos


# Local graph attention
class LocalGraphAttn(nn.Module):
    def __init__(self, d_model: int, d_edge: int = 64, relpos_clip: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.relpos_clip = relpos_clip

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        self.relpos_emb = nn.Embedding(2 * relpos_clip + 1, d_edge)
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_edge, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                nbr_idx: torch.Tensor, nbr_mask: torch.Tensor, rel_pos: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        K = nbr_idx.size(-1)

        # gather neighbors via flatten gather
        flat = x.reshape(B * L, D)
        base = (torch.arange(B, device=x.device) * L).view(B, 1, 1)
        flat_idx = (nbr_idx + base).reshape(-1)
        neigh = flat[flat_idx].reshape(B, L, K, D)  # [B,L,K,D]

        q = self.q(x).unsqueeze(2)            # [B,L,1,D]
        k = self.k(neigh)                     # [B,L,K,D]
        v = self.v(neigh)                     # [B,L,K,D]

        attn = (q * k).sum(-1) / math.sqrt(D)  # [B,L,K]

        rel = torch.clamp(rel_pos, -self.relpos_clip, self.relpos_clip) + self.relpos_clip
        e = self.relpos_emb(rel)
        attn = attn + self.edge_mlp(e).squeeze(-1)

        attn_mask = nbr_mask * mask.unsqueeze(-1)
        w = masked_softmax(attn, attn_mask, dim=-1)

        agg = (w.unsqueeze(-1) * v).sum(dim=2)
        agg = self.out(self.drop(agg))

        g = self.gate(torch.cat([x, agg], dim=-1))
        y = x + g * agg
        y = y * mask.unsqueeze(-1).to(y.dtype)
        return y


class LinearAttention(nn.Module):
    """
    Multi-head linear attention
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def phi(self, z: torch.Tensor) -> torch.Tensor:
        return F.elu(z) + 1.0

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,L,d]
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        q = self.phi(q)
        k = self.phi(k)

        m = mask.view(B, 1, L, 1).to(x.dtype)
        k = k * m
        v = v * m

        kv = torch.einsum("bhld,bhlm->bhdm", k, v)  # [B,H,d,d]
        z = 1.0 / (torch.einsum("bhld,bhd->bhl", q, k.sum(dim=2)) + 1e-6)
        z = z.unsqueeze(-1)

        y = torch.einsum("bhld,bhdm->bhlm", q, kv) * z
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.out(self.drop(y))
        y = y * mask.unsqueeze(-1).to(y.dtype)
        return y


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        rnd = x.new_empty(shape).bernoulli_(keep)
        return x * rnd / keep


class HybridGPSBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.local = LocalGraphAttn(d_model, dropout=dropout)
        self.global_attn = LinearAttention(d_model, n_heads=n_heads, dropout=dropout)

        self.fuse_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.drop_path = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                nbr_idx: torch.Tensor, nbr_mask: torch.Tensor, rel_pos: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)

        h_local = self.local(h, mask, nbr_idx, nbr_mask, rel_pos)
        h_global = h + self.global_attn(h, mask)

        w = torch.softmax(self.fuse_gate(h), dim=-1)
        y = w[..., 0:1] * h_local + w[..., 1:2] * h_global

        x = x + self.drop_path(y - x)  

        x = x + self.drop_path(self.ffn(self.norm2(x)))
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


# TokenLearner pooling
class TokenLearner(nn.Module):
    def __init__(self, d_model: int, m_tokens: int, dropout: float = 0.1):
        super().__init__()
        self.m_tokens = m_tokens
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, m_tokens)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(x)  # [B,L,M]
        attn = masked_softmax(scores, mask.unsqueeze(-1), dim=1)
        y = torch.einsum("blm,bld->bmd", attn, x)
        ymask = x.new_ones((x.size(0), self.m_tokens), dtype=torch.long)
        return y, ymask


# MoE mixer
class ExpertMixer(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, kernel_size: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        conv = self.dwconv(h.transpose(1, 2)).transpose(1, 2)
        y = x + conv
        y = y + self.ffn(self.norm(y))
        y = y * mask.unsqueeze(-1).to(y.dtype)
        return y


class MoEMixer(nn.Module):
    def __init__(self, d_model: int, n_experts: int = 4, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([ExpertMixer(d_model, dropout=dropout) for _ in range(n_experts)])
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_experts)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
        g = (x * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B,D]
        logits = self.gate(g)  # [B,E]

        topv, topi = torch.topk(logits, k=self.top_k, dim=-1)
        weights = torch.softmax(topv, dim=-1)  # [B,top_k]

        out = torch.zeros_like(x)
        for kk in range(self.top_k):
            ei = topi[:, kk]
            w = weights[:, kk].view(-1, 1, 1).to(x.dtype)
            for e in range(self.n_experts):
                sel = (ei == e)
                if sel.any():
                    ye = self.experts[e](x[sel], mask[sel])
                    out[sel] = out[sel] + w[sel] * ye

        out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


# Perceiver latent readout
class PerceiverReadout(nn.Module):
    def __init__(self, d_model: int, n_latents: int = 32, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        kpm = (mask == 0)  # True -> ignore
        y, _ = self.attn(lat, x, x, key_padding_mask=kpm)
        y = self.norm(y)
        return y.mean(dim=1)


@dataclass
class ModelConfig:
    in_dim: int = 1280
    d_model: int = 256
    n_heads: int = 8
    n_blocks: int = 6
    knn_k: int = 32
    knn_local: int = 2
    knn_max_tokens: int = 2500
    pool1_m: int = 128
    pool2_m: int = 32
    moe_experts: int = 4
    moe_layers: int = 2
    readout_latents: int = 32
    dropout: float = 0.1
    drop_path: float = 0.1


class Druggability_DistillModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Sequential(
            nn.LayerNorm(cfg.in_dim),
            nn.Linear(cfg.in_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout)
        )

        dpr = [cfg.drop_path * i / max(1, cfg.n_blocks - 1) for i in range(cfg.n_blocks)]
        self.blocks = nn.ModuleList([
            HybridGPSBlock(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, drop_path=dpr[i])
            for i in range(cfg.n_blocks)
        ])

        self.pool1 = TokenLearner(cfg.d_model, cfg.pool1_m, dropout=cfg.dropout)
        self.pool2 = TokenLearner(cfg.d_model, cfg.pool2_m, dropout=cfg.dropout)

        self.moe = nn.ModuleList([
            MoEMixer(cfg.d_model, n_experts=cfg.moe_experts, top_k=2, dropout=cfg.dropout)
            for _ in range(cfg.moe_layers)
        ])

        self.readout = PerceiverReadout(cfg.d_model, n_latents=cfg.readout_latents, n_heads=cfg.n_heads, dropout=cfg.dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_tokens: bool = False) -> Dict[str, torch.Tensor]:
        # x: [B,L,1280]
        x = self.in_proj(x)

        # dynamic knn graph on current representation
        nbr_idx, nbr_mask, rel_pos = build_dynamic_knn(
            x, mask, k=self.cfg.knn_k, add_local=self.cfg.knn_local, max_tokens_for_knn=self.cfg.knn_max_tokens
        )

        for blk in self.blocks:
            x = blk(x, mask, nbr_idx, nbr_mask, rel_pos)

        # hierarchical compression
        x1, m1 = self.pool1(x, mask)
        x2, m2 = self.pool2(x1, m1)

        # MoE long mixer
        for moe in self.moe:
            x2 = moe(x2, m2)

        z = self.readout(x2, m2)
        logit = self.head(z).squeeze(-1)

        out = {"logit": logit}
        if return_tokens:
            out["tokens"] = x2
        return out


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(decay).add_(s.data, alpha=1.0 - decay)

    for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
        t_buf.copy_(s_buf)


class TeacherPostProcess:
    """
    Maintain an EMA center of teacher logits, then:
      centered_logit = t_logit - center
      pt = sigmoid(centered_logit / temp)
    """
    def __init__(self, momentum: float = 0.9, temp: float = 0.7):
        self.m = momentum
        self.temp = temp
        self.center = 0.0

    @torch.no_grad()
    def update_center(self, teacher_logit: torch.Tensor):
        # teacher_logit: [B]
        c = float(teacher_logit.mean().item())
        self.center = self.m * self.center + (1.0 - self.m) * c

    @torch.no_grad()
    def prob(self, teacher_logit: torch.Tensor) -> torch.Tensor:
        centered = teacher_logit - self.center
        return torch.sigmoid(centered / self.temp)


# Losses
def bce_with_logits(logit: torch.Tensor, y: torch.Tensor, pos_weight: Optional[float] = None) -> torch.Tensor:
    if pos_weight is not None:
        pw = torch.tensor(pos_weight, device=logit.device, dtype=logit.dtype)
        return F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)
    return F.binary_cross_entropy_with_logits(logit, y)


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def kd_loss_binary(student_logit: torch.Tensor,
                   teacher_prob: torch.Tensor,
                   T: float = 2.0,
                   conf_weight: bool = True) -> torch.Tensor:
    """
    AMP-safe KD:
      use BCEWithLogits on (student_logit / T) against teacher_prob
    """
    logits = student_logit / T
    pt = teacher_prob.detach().clamp(1e-6, 1.0 - 1e-6)

    loss = F.binary_cross_entropy_with_logits(logits, pt) * (T * T)

    if conf_weight:
        with torch.no_grad():
            ent = binary_entropy(pt.float())
            w = (1.0 - ent / math.log(2.0)).mean()   # [0,1]
        loss = loss * w.to(loss.dtype)
    return loss


def relational_kd_loss(student_tokens: torch.Tensor, teacher_tokens: torch.Tensor, n_sample: int = 32) -> torch.Tensor:
    """
    Align token-token similarity structure on compressed tokens.
    """
    B, N, D = student_tokens.shape
    ns = min(n_sample, N)
    idx = torch.randperm(N, device=student_tokens.device)[:ns]
    s = F.normalize(student_tokens[:, idx], dim=-1)
    t = F.normalize(teacher_tokens[:, idx], dim=-1)
    Ss = s @ s.transpose(1, 2)
    St = t @ t.transpose(1, 2)
    return F.mse_loss(Ss, St)


def rdrop_kl(logit1: torch.Tensor, logit2: torch.Tensor) -> torch.Tensor:
    """
    R-Drop consistency for binary:
      KL(p1||p2) + KL(p2||p1) where p=sigmoid(logit)
    """
    p1 = torch.sigmoid(logit1).clamp(1e-6, 1 - 1e-6)
    p2 = torch.sigmoid(logit2).clamp(1e-6, 1 - 1e-6)

    kl12 = p1 * (torch.log(p1) - torch.log(p2)) + (1 - p1) * (torch.log(1 - p1) - torch.log(1 - p2))
    kl21 = p2 * (torch.log(p2) - torch.log(p1)) + (1 - p2) * (torch.log(1 - p2) - torch.log(1 - p1))
    return (kl12 + kl21).mean()


@torch.no_grad()
def roc_auc_score_torch(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    y_true: [N] 0/1
    y_score: [N] probability
    """
    y_true = y_true.detach().cpu().float()
    y_score = y_score.detach().cpu().float()
    n = y_true.numel()
    if n == 0:
        return float("nan")
    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(y_score)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float)  # 1..N

    sum_ranks_pos = ranks[pos].sum().item()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def cosine_warmup_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses = []
    ys = []
    ps = []

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y = batch["y"].to(device)

        out = model(x, mask, return_tokens=False)
        logit = out["logit"]
        loss = F.binary_cross_entropy_with_logits(logit, y)

        p = torch.sigmoid(logit)
        losses.append(loss.detach())
        ys.append(y.detach())
        ps.append(p.detach())

    loss = torch.stack(losses).mean().item()
    y_all = torch.cat(ys, dim=0)
    p_all = torch.cat(ps, dim=0)

    acc = ((p_all >= 0.5).float() == y_all).float().mean().item()
    auc = roc_auc_score_torch(y_all, p_all)
    return {"loss": loss, "acc": acc, "auc": auc}


def train(args):
    seed_all(args.seed)
    device = torch.device(args.device)

    # datasets
    train_ds = ProteinEmbeddingDataset(args.base_dir, "Train")
    test_ds  = ProteinEmbeddingDataset(args.base_dir, "Test")

    idx = list(range(len(train_ds)))
    random.shuffle(idx)
    n_val = int(len(idx) * args.val_ratio)
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    tr_set = torch.utils.data.Subset(train_ds, tr_idx)
    va_set = torch.utils.data.Subset(train_ds, val_idx)

    tr_loader = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                           collate_fn=collate_fn, pin_memory=True)
    va_loader = DataLoader(va_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                           collate_fn=collate_fn, pin_memory=True)
    ts_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                           collate_fn=collate_fn, pin_memory=True)

    # model
    cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        knn_k=args.knn_k,
        knn_local=args.knn_local,
        knn_max_tokens=args.knn_max_tokens,
        pool1_m=args.pool1,
        pool2_m=args.pool2,
        moe_experts=args.moe_experts,
        moe_layers=args.moe_layers,
        readout_latents=args.readout_latents,
        dropout=args.dropout,
        drop_path=args.drop_path
    )

    student = Druggability_DistillModel(cfg).to(device)
    teacher = Druggability_DistillModel(cfg).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    teacher_pp = TeacherPostProcess(momentum=args.teacher_center_m, temp=args.teacher_sharpen_temp)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    pos_weight = None
    if args.auto_pos_weight:
        y_tmp = []
        for i in range(min(2000, len(tr_set))):
            y_tmp.append(tr_set[i]["label"])
        pos = sum(y_tmp)
        neg = len(y_tmp) - pos
        if pos > 0:
            pos_weight = float(neg / max(1, pos))
            print(f"[INFO] Auto pos_weight ≈ {pos_weight:.3f} (neg/pos)")

    total_steps = args.epochs * max(1, len(tr_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_auc = -1.0
    global_step = 0

    print(f"[INFO] Train samples: {len(tr_set)}, Val samples: {len(va_set)}, Test samples: {len(test_ds)}")
    print(f"[INFO] total_steps={total_steps}, warmup_steps={warmup_steps}")

    for ep in range(1, args.epochs + 1):
        student.train()
        t0 = time.time()

        sup_meter = 0.0
        kd_meter = 0.0
        rel_meter = 0.0
        rdrop_meter = 0.0
        n_seen = 0

        for batch in tr_loader:
            global_step += 1

            # lr schedule
            lr_now = cosine_warmup_lr(global_step, total_steps, args.lr, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            x = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with torch.no_grad():
                x_t = make_global_view(x, mask, noise_std=args.t_global_noise, token_dropout=args.t_global_drop)
                out_t = teacher(x_t, mask, return_tokens=True)
                t_logit = out_t["logit"]
                t_tokens = out_t["tokens"]

                teacher_pp.update_center(t_logit)
                t_prob = teacher_pp.prob(t_logit)  

            # student global view
            x_sg = make_global_view(x, mask, noise_std=args.s_global_noise, token_dropout=args.s_global_drop)

            # local crops
            locals_xm: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for _ in range(args.n_local_crops):
                xl, ml = make_local_view(
                    x, mask,
                    crop_ratio_min=args.local_crop_min,
                    crop_ratio_max=args.local_crop_max,
                    noise_std=args.local_noise,
                    token_dropout=args.local_drop
                )
                locals_xm.append((xl.to(device, non_blocking=True), ml.to(device, non_blocking=True)))

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                out_sg = student(x_sg, mask, return_tokens=True)
                s_logit_g = out_sg["logit"]
                s_tokens_g = out_sg["tokens"]

                sup = bce_with_logits(s_logit_g, y, pos_weight=pos_weight)

                # KD on global
                kd_g = kd_loss_binary(s_logit_g, t_prob, T=args.kd_T, conf_weight=True)
                sup_local = 0.0
                kd_local = 0.0
                for (xl, ml) in locals_xm:
                    out_sl = student(xl, ml, return_tokens=False)
                    s_logit_l = out_sl["logit"]
                    sup_local = sup_local + bce_with_logits(s_logit_l, y, pos_weight=pos_weight)
                    kd_local = kd_local + kd_loss_binary(s_logit_l, t_prob, T=args.kd_T, conf_weight=True)

                if args.n_local_crops > 0:
                    sup_local = sup_local / args.n_local_crops
                    kd_local = kd_local / args.n_local_crops
                else:
                    sup_local = torch.tensor(0.0, device=device)
                    kd_local = torch.tensor(0.0, device=device)

                rel = relational_kd_loss(s_tokens_g, t_tokens, n_sample=args.rel_kd_sample) if args.rel_kd_w > 0 else torch.tensor(0.0, device=device)

                if args.rdrop_w > 0:
                    out_sg2 = student(x_sg, mask, return_tokens=False)
                    rkl = rdrop_kl(s_logit_g, out_sg2["logit"])
                else:
                    rkl = torch.tensor(0.0, device=device)

                loss = (
                    sup
                    + args.sup_local_w * sup_local
                    + args.kd_w * (kd_g + kd_local)
                    + args.rel_kd_w * rel
                    + args.rdrop_w * rkl
                )

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ema_update(teacher, student, decay=args.ema_decay)

            bs = x.size(0)
            n_seen += bs
            sup_meter += float(sup.detach().item()) * bs
            kd_meter += float((kd_g + kd_local).detach().item()) * bs
            rel_meter += float(rel.detach().item()) * bs
            rdrop_meter += float(rkl.detach().item()) * bs

        tr_time = time.time() - t0
        val_t = evaluate(teacher, va_loader, device=device)
        val_s = evaluate(student, va_loader, device=device)
        tst_t = evaluate(teacher, ts_loader, device=device)
        tst_s = evaluate(student, ts_loader, device=device)

        print(
            f"[Epoch {ep:03d}] "
            f"train_sup={sup_meter/max(1,n_seen):.4f} "
            f"train_kd={kd_meter/max(1,n_seen):.4f} "
            f"train_rel={rel_meter/max(1,n_seen):.4f} "
            f"train_rdrop={rdrop_meter/max(1,n_seen):.4f} "
            f"| teacher_val_loss={val_t['loss']:.4f} val_acc={val_t['acc']:.4f} val_auc={val_t['auc']:.4f} "
            f"| teacher_test_loss={tst_t['loss']:.4f} test_acc={tst_t['acc']:.4f} test_auc={tst_t['auc']:.4f} "
            
            f"| student_val_loss={val_s['loss']:.4f} val_acc={val_s['acc']:.4f} val_auc={val_s['auc']:.4f} "
            f"| student_test_loss={tst_s['loss']:.4f} test_acc={tst_s['acc']:.4f} test_auc={tst_s['auc']:.4f} "
            
            f"| time={tr_time:.1f}s"
        )

        # save best val
        if not math.isnan(val_t["auc"]) and val_t["auc"] > best_val_auc:
            best_val_auc = val_t["auc"]
            ckpt_path = os.path.join(args.out_dir, "best_ema_teacher.pt")
            torch.save({"cfg": cfg.__dict__, "state_dict": teacher.state_dict()}, ckpt_path)
            print(f"[SAVE] best EMA teacher -> {ckpt_path} (val_auc={best_val_auc:.4f})")

    final_path = os.path.join(args.out_dir, "final_ema_teacher.pt")
    torch.save({"cfg": cfg.__dict__, "state_dict": teacher.state_dict()}, final_path)
    print(f"[DONE] final EMA teacher saved -> {final_path}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_dir", type=str, default="../data", help="Base directory containing Train_/Test_ folders")
    ap.add_argument("--out_dir", type=str, default="./results", help="Output directory for checkpoints")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # dataloader
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    # training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--auto_pos_weight", action="store_true")

    # model knobs
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--drop_path", type=float, default=0.10)

    ap.add_argument("--knn_k", type=int, default=16)
    ap.add_argument("--knn_local", type=int, default=2)
    ap.add_argument("--knn_max_tokens", type=int, default=500)

    ap.add_argument("--pool1", type=int, default=128)
    ap.add_argument("--pool2", type=int, default=32)

    ap.add_argument("--moe_experts", type=int, default=4)
    ap.add_argument("--moe_layers", type=int, default=2)
    ap.add_argument("--readout_latents", type=int, default=32)

    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--teacher_center_m", type=float, default=0.9)
    ap.add_argument("--teacher_sharpen_temp", type=float, default=0.7)

    ap.add_argument("--kd_w", type=float, default=0.7)
    ap.add_argument("--kd_T", type=float, default=2.0)
    ap.add_argument("--rel_kd_w", type=float, default=0.1)
    ap.add_argument("--rel_kd_sample", type=int, default=32)

    ap.add_argument("--rdrop_w", type=float, default=0.1)
    ap.add_argument("--n_local_crops", type=int, default=1)
    ap.add_argument("--sup_local_w", type=float, default=0.2)

    ap.add_argument("--t_global_noise", type=float, default=0.01)
    ap.add_argument("--t_global_drop", type=float, default=0.05)
    ap.add_argument("--s_global_noise", type=float, default=0.01)
    ap.add_argument("--s_global_drop", type=float, default=0.05)

    ap.add_argument("--local_crop_min", type=float, default=0.4)
    ap.add_argument("--local_crop_max", type=float, default=0.8)
    ap.add_argument("--local_noise", type=float, default=0.02)
    ap.add_argument("--local_drop", type=float, default=0.10)

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()