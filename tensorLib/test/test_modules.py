import pytest
# import tensor_bindings as tb  # the module is a .so file compiled from C++
# from tensorLib.build import tensor_bindings as tb
# import numpy as np
import random
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('/raid/home/zhuyangyang/tensorLib/build')
import tensor_bindings as tb
import time

def test_rmsnorm():
    dim = 4096
    num_tokens = 1024
    # device = "cpu"
    device = "cuda"

    # define the RMSNorm layer
    rms_norm_torch = nn.RMSNorm(dim, device=device)
    rms_norm_tb = tb.RMSNorm(dim, device=device)

    # define the input tensor
    input_torch = torch.randn(num_tokens, dim, device=device)
    input_tb = tb.convert_to_tensor(input_torch.cpu().numpy(), device)

    # set the weight of the RMSNorm layer as the same
    weight_torch = torch.randn(dim, device=device)
    weight_tb = tb.convert_to_tensor(weight_torch.detach().cpu().numpy(), device)
    rms_norm_torch.weight = torch.nn.Parameter(weight_torch)
    rms_norm_tb.weight = weight_tb

    # inference of three methods
    with torch.no_grad():
        start_time = time.time()
        result0 = rms_norm_torch.forward(input_torch)
        end_time = time.time()
        print(f"Execution time for PyTorch RMSNorm: {end_time - start_time} seconds")

    start_time = time.time()
    result1 = rms_norm_tb.forward(input_tb)
    end_time = time.time()
    print(f"Execution time for tensor_bindings RMSNorm: {end_time - start_time} seconds")
    result1_np = tb.convert_to_numpy(result1)

    if device == "cuda":
        start_time = time.time()
        result2 = rms_norm_tb.forward_fused_cuda(input_tb)
        end_time = time.time()
        print(f"Execution time for tensor_bindings RMSNorm fused CUDA: {end_time - start_time} seconds")
        result2_np = tb.convert_to_numpy(result2)

    # check the correctness
    assert result0.cpu().numpy().shape == result1_np.shape
    assert result0.cpu().numpy().dtype == result1_np.dtype
    assert result0.cpu().numpy().size == result1_np.size
    assert np.allclose(result0.cpu().numpy(), result1_np, rtol=1e-4, atol=1e-4)

    if device == "cuda":
        assert result0.cpu().numpy().shape == result2_np.shape
        assert result0.cpu().numpy().dtype == result2_np.dtype
        assert result0.cpu().numpy().size == result2_np.size
        assert np.allclose(result0.cpu().numpy(), result2_np, rtol=1e-3, atol=1e-4)

# main
if __name__ == "__main__":
    test_rmsnorm()
    print("All tests passed!")
