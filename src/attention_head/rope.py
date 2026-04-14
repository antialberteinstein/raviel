import torch

from config import config


def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    """Tạo ra các góc xoay (frequencies) cho từng đơn vị trong câu"""

    device = config.device

    # Tính toán góc xoay cho từng cặp chiều không gian
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)  # Mảng vị trí

    # Nhân ma trận vị trí với góc xoay
    freqs - torch.outer(t, freqs)  # (seq_len, dim/2)

    # Lấy SIn và Cos
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    return freqs_cos.to(device), freqs_sin.to(device)


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    """Áp dụng phép xoay hình học cho Query và Key"""
    B, T, C = xq.shape

    # Tách vector thành các cặp chẵn/lẻ để xoay 2D
    xq_even = xq[..., 0::2]
    xq_odd = xq[..., 1::2]
    xk_even = xk[..., 0::2]
    xk_odd = xk[..., 1::2]

    # Áp dụng công thức xoay ma trận lượng giác
    xq_rotated = torch.empty_like(xq)
    xq_rotated[..., 0::2] = xq_even * freqs_cos - xq_odd * freqs_sin
    xq_rotated[..., 1::2] = xq_odd * freqs_cos + xq_even * freqs_sin

    xk_rotated = torch.empty_like(xk)
    xk_rotated[..., 0::2] = xk_even * freqs_cos - xk_odd * freqs_sin
    xk_rotated[..., 1::2] = xk_odd * freqs_cos + xk_even * freqs_sin

    return xq_rotated, xk_rotated
