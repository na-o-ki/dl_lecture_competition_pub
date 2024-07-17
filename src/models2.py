import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim * 2),
            ConvBlock(hid_dim * 2, hid_dim * 4),
            ConvBlock(hid_dim * 4, hid_dim * 8),
            SelfAttention(hid_dim * 8)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim * 8, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.3,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding=1)
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding=1)

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.layernorm = nn.LayerNorm(out_dim)

        self.dropout = nn.Dropout(p_drop)

        if in_dim != out_dim:
            self.residual_conv = nn.Conv1d(in_dim, out_dim, kernel_size=1, padding=0)
        else:
            self.residual_conv = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        residual = X
        if self.residual_conv is not None:
            residual = self.residual_conv(X)
        
        X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))
        
        X = self.conv1(X) + residual  # skip connection
        X = F.gelu(self.batchnorm1(X))
        
        # Change the shape to apply LayerNorm correctly
        X = X.permute(0, 2, 1)  # [batch_size, seq_len, features]
        X = self.layernorm(X)
        X = X.permute(0, 2, 1)  # [batch_size, features, seq_len]

        return self.dropout(X)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, d, n = x.shape
        h = self.heads
        qkv = self.to_qkv(x.permute(0, 2, 1)).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out).permute(0, 2, 1)
