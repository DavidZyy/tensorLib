import pytest
import random
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
# sys.path.append('/home/zyy/project/tensorLib/build')
# sys.path.append('/raid/home/zhuyangyang/tensorLib/build')
sys.path.append('/root/autodl-tmp/tensorLib/build')
import tensor_bindings as tb
import time

def test_matmul():
    m = 3
    n = 2
    p = 4
    a = torch.randn(m, n)
    b = torch.randn(n, p)

    c = a.matmul(b)
    c = torch.matmul(a, b)

    c_tb = tb.matmul(a, b)
    c_tb_np = tb.convert_to_numpy(c_tb)

    assert c.cpu().numpy().shape == c_tb_np.shape
    assert c.cpu().numpy().dtype == c_tb_np.dtype
    assert c.cpu().numpy().size == c_tb_np.size
    assert np.allclose(c.cpu().numpy(), c_tb_np, rtol=1e-4, atol=1e-4)

def test_argmin():
    A = np.arange(32).astype(np.float32).reshape(1, 32)
    tmp = A[0, 7]
    A[0, 7] = A[0, 31]
    A[0, 31] = tmp
    b = A.argmax(1).astype(np.int32)

    A_t = tb.convert_to_tensor(A, "cuda")
    tensor_op = getattr(A_t, "argmax")
    tensor_result = tensor_op(1, False)
    tensor_result_a = tb.convert_to_numpy(tensor_result)

    np.testing.assert_allclose(b, tensor_result_a)
    print(A)

#######################################################################################################################################
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
 
    Returns:
        torch.Tensor: reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a tensor

    Args:
        x (torch.Tensor): shape (batch_size, seq_len, num_heads, head_dim)
        freqs_cis (torch.Tensor): shape (seq_len, head_dim)

    Returns:
        torch.Tensor: shape (batch_size, seq_len, num_heads, head_dim)
    """
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)

def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0):
    dim = head_dim
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def test_apply_rotary_emb():
    B = 7
    T = 9
    n_heads = 32
    head_dim = 128
    freqs_cis = precompute_freqs_cis(head_dim, 1024)
    x = torch.randn(B, T, n_heads, head_dim)
    x_r = apply_rotary_emb(x, freqs_cis[0: T])

    x_tb = tb.convert_to_tensor(x, "cuda")
    # x_tb = tb.convert_to_tensor(x, "cpu")
    x_tb_r = tb.apply_rotary_emb(x_tb, 0)
    x_tb_r_np = tb.convert_to_numpy(x_tb_r)

    np.testing.assert_allclose(x_r.cpu().numpy(), x_tb_r_np, rtol=1e-4, atol=1e-4)
    pass

#######################################################################################################################################
