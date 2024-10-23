import pytest
import torch
import tensor_bindings as tb
import numpy as np
# import libtensor_bindings as tb
# from . import libtensor_bindings as tb

def test_sum():
    shape = [2, 3]
    
    # PyTorch sum
    a_torch = torch.randn(shape)
    result_torch = a_torch.sum(dim=1)
    
    # C++ tensor sum
    a_cpp = tb.Tensor(shape)
    a_cpp_max = a_cpp.max(0, False)

    a_cpp_numpy = tb.convert_to_numpy(a_cpp)
    a_cpp_max_numpy = tb.convert_to_numpy(a_cpp_max)

    # Compare results
    assert torch.allclose(result_torch, torch.tensor(result_cpp.data())), "Sum result mismatch!"
