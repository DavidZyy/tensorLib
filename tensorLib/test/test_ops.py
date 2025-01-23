import pytest
import random
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('/home/zyy/project/tensorLib/build')
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
