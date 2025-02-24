import pytest
import random
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
# sys.path.append('/raid/home/zhuyangyang/tensorLib/build')
# sys.path.append('../../build')
# sys.path.append('/home/zyy/project/tensorLib/build')
sys.path.append('/root/gpufree-data/tensorLib/build')
import tensor_bindings as tb
import time

def test_rmsnorm():
    dim = 1024
    num_tokens = 128
    # dim = 128
    # num_tokens = 4
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

    # inference of methods
    with torch.no_grad():
        start_time = time.time()
        result0 = rms_norm_torch.forward(input_torch)
        end_time = time.time()
        print(f"Execution time for PyTorch RMSNorm: {end_time - start_time} seconds")

    start_time = time.time()
    result1 = rms_norm_tb.forward_plain(input_tb)
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
    # assert np.allclose(result0.cpu().numpy(), result1_np, rtol=1e-4, atol=1e-4)
    assert np.allclose(result0.cpu().numpy(), result1_np, rtol=0, atol=1e-4)

    if device == "cuda":
        assert result0.cpu().numpy().shape == result2_np.shape
        assert result0.cpu().numpy().dtype == result2_np.dtype
        assert result0.cpu().numpy().size == result2_np.size
        # assert np.allclose(result0.cpu().numpy(), result2_np, rtol=1e-4, atol=1e-4)
        assert np.allclose(result0.cpu().numpy(), result2_np, rtol=0, atol=1e-4)

def test_relu():
    dim = 4096
    num_tokens = 1024
    device = "cuda"

    # define the ReLU layer
    relu_torch = nn.ReLU()
    relu_tb = tb.ReLU()

    # define the input tensor
    input_torch = torch.randn(num_tokens, dim, device=device)
    input_tb = tb.convert_to_tensor(input_torch.cpu().numpy(), device)

    # inference of methods
    with torch.no_grad():
        start_time = time.time()
        result0 = relu_torch.forward(input_torch)
        end_time = time.time()
        print(f"Execution time for PyTorch ReLU: {end_time - start_time} seconds")

    start_time = time.time()
    result1 = relu_tb.forward(input_tb)
    end_time = time.time()
    print(f"Execution time for tensor_bindings ReLU: {end_time - start_time} seconds")
    result1_np = tb.convert_to_numpy(result1)

    assert result0.cpu().numpy().shape == result1_np.shape
    assert result0.cpu().numpy().dtype == result1_np.dtype
    assert result0.cpu().numpy().size == result1_np.size
    assert np.allclose(result0.cpu().numpy(), result1_np, rtol=1e-4, atol=1e-4)

def test_conv2d():
    in_channels = 3
    out_channels = 5
    kernel_size = 3
    stride = 1
    padding = 1
    device = "cuda"

    # define the Conv2d layer
    conv2d_torch = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    conv2d_tb = tb.Conv2d(in_channels, out_channels, kernel_size, stride, padding, device=device)

    # define the input tensor
    input_torch = torch.randn(1, in_channels, 5, 5, device=device)
    input_tb = tb.convert_to_tensor(input_torch.cpu().numpy(), device)

    # set the weight of the Conv2d layer as the same
    weight_torch = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device)
    weight_tb = tb.convert_to_tensor(weight_torch.detach().cpu().numpy(), device)
    conv2d_torch.weight = torch.nn.Parameter(weight_torch)
    conv2d_tb.weight = weight_tb

    # inference of methods
    with torch.no_grad():
        start_time = time.time()
        result0 = conv2d_torch.forward(input_torch)
        end_time = time.time()
        print(f"Execution time for PyTorch Conv2d: {end_time - start_time} seconds")

    start_time = time.time()
    result1 = conv2d_tb.forward(input_tb)
    end_time = time.time()
    print(f"Execution time for tensor_bindings Conv2d: {end_time - start_time} seconds")
    result1_np = tb.convert_to_numpy(result1)

    assert result0.cpu().numpy().shape == result1_np.shape
    assert result0.cpu().numpy().dtype == result1_np.dtype
    assert result0.cpu().numpy().size == result1_np.size
    assert np.allclose(result0.cpu().numpy(), result1_np, rtol=1e-4, atol=1e-4)

def test_embedding():
    num_embeddings = 10
    embedding_dim = 3
    device = "cuda"

    # define the Embedding layer
    embedding_torch = nn.Embedding(num_embeddings, embedding_dim)
    embedding_tb = tb.Embedding(num_embeddings, embedding_dim, device=device)

    # define the input tensor
    input_torch = torch.tensor([1, 2, 3, 4, 5], device=device)
    input_tb = tb.convert_to_tensor(input_torch.cpu().numpy(), device)

    # set the weight of the Embedding layer as the same
    weight_torch = torch.randn(num_embeddings, embedding_dim, device=device)
    weight_tb = tb.convert_to_tensor(weight_torch.detach().cpu().numpy(), device)
    embedding_torch.weight = torch.nn.Parameter(weight_torch)
    embedding_tb.weight = weight_tb

    # inference of methods
    with torch.no_grad():
        start_time = time.time()
        result0 = embedding_torch.forward(input_torch)
        end_time = time.time()
        print(f"Execution time for PyTorch Embedding: {end_time - start_time} seconds")

    start_time = time.time()
    result1 = embedding_tb.forward(input_tb)
    end_time = time.time()
    print(f"Execution time for tensor_bindings Embedding: {end_time - start_time} seconds")
    result1_np = tb.convert_to_numpy(result1)

    assert result0.cpu().numpy().shape == result1_np.shape
    assert result0.cpu().numpy().dtype == result1_np.dtype
    assert result0.cpu().numpy().size == result1_np.size
    assert np.allclose(result0.cpu().numpy(), result1_np, rtol=1e-4, atol=1e-4)

def test_moduleList():
    num_modules = 5
    in_channels = 3
    out_channels = 5
    kernel_size = 3
    stride = 1
    padding = 1
    device = "cuda"

    # define the Conv2d layer
    conv2d_torch = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    conv2d_tb = tb.Conv2d(in_channels, out_channels, kernel_size, stride, padding, device=device)

    # define the ModuleList
    module_list_torch = nn.ModuleList([conv2d_torch for _ in range(num_modules)])
    module_list_tb = tb.ModuleList([conv2d_tb for _ in range(num_modules)])

    # define the input tensor
    input_torch = torch.randn(1, in_channels, 5, 5, device=device)
    input_tb = tb.convert_to_tensor(input_torch.cpu().numpy(), device)

    # set the weight of the Conv2d layer as the same
    weight_torch = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device)
    weight_tb = tb.convert_to_tensor(weight_torch.detach().cpu().numpy(), device)
    conv2d_torch.weight = torch.nn.Parameter(weight_torch)
    conv2d_tb.weight = weight_tb

    # inference of methods
    with torch.no_grad():
        start_time = time.time()
        result0 = module_list_torch[0].forward(input_torch)
        end_time = time.time()
        print(f"Execution time for PyTorch ModuleList: {end_time - start_time} seconds")

    start_time = time.time()
    result1 = module_list_tb[0].forward(input_tb)
    end_time = time.time()
    print(f"Execution time for tensor_bindings ModuleList: {end_time - start_time} seconds")
    result1_np = tb.convert_to_numpy(result1)

    assert result0.cpu().numpy().shape == result1_np.shape
    assert result0.cpu().numpy().dtype == result1_np.dtype
    assert result0.cpu().numpy().size == result1_np.size
    assert np.allclose(result0.cpu().numpy(), result1_np, rtol=1e-4, atol=1e-4)

def test_linear():
    in_features = 3
    out_features = 5
    device = "cuda"

    # define the Linear layer
    linear_torch = nn.Linear(in_features, out_features)
    linear_tb = tb.Linear(in_features, out_features, device=device)

    # define the input tensor
    input_torch = torch.randn(1, in_features, device=device)
    input_tb = tb.convert_to_tensor(input_torch.cpu().numpy(), device)

    # set the weight of the Linear layer as the same
    weight_torch = torch.randn(out_features, in_features, device=device)
    weight_tb = tb.convert_to_tensor(weight_torch.detach().cpu().numpy(), device)
    linear_torch.weight = torch.nn.Parameter(weight_torch)
    linear_tb.weight = weight_tb

    # inference of methods
    with torch.no_grad():
        start_time = time.time()
        result0 = linear_torch.forward(input_torch)
        end_time = time.time()
        print(f"Execution time for PyTorch Linear: {end_time - start_time} seconds")

    start_time = time.time()
    result1 = linear_tb.forward(input_tb)
    end_time = time.time()
    print(f"Execution time for tensor_bindings Linear: {end_time - start_time} seconds")
    result1_np = tb.convert_to_numpy(result1)

    assert result0.cpu().numpy().shape == result1_np.shape
    assert result0.cpu().numpy().dtype == result1_np.dtype
    assert result0.cpu().numpy().size == result1_np.size
    assert np.allclose(result0.cpu().numpy(), result1_np, rtol=1e-4, atol=1e-4)

# main
if __name__ == "__main__":
    test_rmsnorm()
    print("All tests passed!")
