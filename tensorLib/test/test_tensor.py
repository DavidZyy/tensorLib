import pytest
import tensor_bindings as tb  # the module is a .so file compiled from C++
import numpy as np
import random
# import libtensor_bindings as tb
# from . import libtensor_bindings as tb


def generate_random_shapes(n_shapes, min_dims=0, max_dims=4, max_size=10):
    """
    Generate a list of random shapes, including the possibility of scalar shape ().
    
    :param n_shapes: Number of shapes to generate.
    :param max_dims: Maximum number of dimensions a shape can have.
    :param max_size: Maximum size for each dimension.
    :return: List of randomly generated shapes.
    """
    shapes = []
    for _ in range(n_shapes):
        # Randomly decide the number of dimensions (0 for scalar, up to max_dims)
        num_dims = random.randint(min_dims, max_dims)
        if num_dims == 0:
            shapes.append(())  # Scalar shape
        else:
            # Generate a shape with random sizes for each dimension
            shape = tuple(random.randint(1, max_size) for _ in range(num_dims))
            shapes.append(shape)
    
    return shapes


convert_shapes = generate_random_shapes(100, min_dims=0, max_dims=4, max_size=100)


@pytest.mark.parametrize("shape", convert_shapes)
def test_convert(shape):
    # print(shape)
    # Generate random data
    A = np.random.randn(*shape)
    # A = np.array([-1.234])

    # If shape is (), convert A to a NumPy array to ensure it's not a mere float
    if shape == ():
        A = np.array(A, dtype=np.float32)
    else:
        A = A.astype(np.float32)

    A_t = tb.convert_to_tensor(A)
    A_t_a = tb.convert_to_numpy(A_t)
    
    # print(A)
    # print(A_t)
    assert A.shape == A_t_a.shape
    assert A.dtype == A_t_a.dtype
    assert A.size == A_t_a.size
    np.testing.assert_allclose(A, A_t_a, atol=1e-5, rtol=1e-5)


def generate_batched_matmul_shapes(batch_size_range=(1, 10), dim_range=(1, 100)):
    """
    Generate valid shape1 and shape2 for batched matrix multiplication with 1 or 2 batch dimensions.
    
    :param batch_size_range: Range for possible batch sizes (number of elements in batch dimensions).
    :param dim_range: Range for matrix dimensions.
    :return: A tuple containing (shape1, shape2) for valid batched matrix multiplication.

    Batch Matrix Multiplication Shape Rules:
    1. The first N-2 dimensions are considered as batch dimensions and must be broadcastable.
    2. The last two dimensions of the input tensors should be compatible for matrix multiplication:
            For shape1 (A), the second-to-last dimension represents the number of rows.
            For shape2 (B), the last dimension represents the number of columns.
            The number of columns of A (i.e., shape1[-1]) must match the number of rows of B (i.e., shape2[-2]).
    """
    # Randomly select 1 or 2 for the number of batch dimensions
    num_batch_dims = random.choice([1, 2])

    # Randomly generate batch sizes for either 1 or 2 batch dimensions
    batch_shape = tuple(random.randint(batch_size_range[0], batch_size_range[1]) for _ in range(num_batch_dims))

    # Randomly select dimensions for shape1 and shape2
    m = random.randint(dim_range[0], dim_range[1])  # Rows in shape1
    k = random.randint(dim_range[0], dim_range[1])  # Shared dimension (columns in shape1, rows in shape2)
    n = random.randint(dim_range[0], dim_range[1])  # Columns in shape2

    # Construct shape1 and shape2 with batch dimensions
    shape1 = batch_shape + (m, k)   # (batch_shape..., m, k) for the first matrix
    shape2 = batch_shape + (k, n)   # (batch_shape..., k, n) for the second matrix

    return shape1, shape2

bached_matmul_shapes = [generate_batched_matmul_shapes() for _ in range(50)]

@pytest.mark.parametrize("shape1, shape2", bached_matmul_shapes)
def test_batched_matmul(shape1, shape2):
    A = np.random.randn(*shape1).astype(np.float32)  # must convert to float32!!! or it will be float64!!
    B = np.random.randn(*shape2).astype(np.float32)
    C = np.matmul(A, B)

    A_t = tb.convert_to_tensor(A)
    B_t = tb.convert_to_tensor(B)
    C_t = A_t.matmul(B_t)
    C_t_a = tb.convert_to_numpy(C_t)

    assert C.shape == C_t_a.shape
    assert C.dtype == C_t_a.dtype
    assert C.size == C_t_a.size
    np.testing.assert_allclose(C, C_t_a, atol=1e-5, rtol=1e-5)


reduced_ops = [
    ('sum', np.sum),
    ('mean', np.mean),
    ('max', np.max),
    ('min', np.min),
    ('argmax', np.argmax),
    ('argmin', np.argmin)
]

@pytest.mark.parametrize("shape", generate_random_shapes(50, min_dims=1, max_dims=2, max_size=5))
@pytest.mark.parametrize("op, np_op", reduced_ops)
@pytest.mark.parametrize("keepdims", [True, False])
def test_reduced_methods(shape, op, np_op, keepdims):
    """
    Test the reduced methods, which are sum, mean, max, min, argmax, argmin...
    """
    assert len(shape) > 0   # shape != ()

    # Generate random tensor data
    A = np.random.randn(*shape).astype(np.float32)
    axis = random.choice(range(len(shape)))
    np_result = np_op(A, axis=axis, keepdims=keepdims)
    if op == 'argmax' or op == 'argmin':
        np_result = np_result.astype(np.int32)

    # Convert NumPy array to Tensor
    A_t = tb.convert_to_tensor(A)
    tensor_op = getattr(A_t, op)
    tensor_result = tensor_op(axis, keepdims)
    tensor_result_a = tb.convert_to_numpy(tensor_result)

    print(A)
    print(np_result)
    print(tensor_result_a)
    # Assert that shapes are equal
    assert tensor_result_a.shape == np_result.shape
    assert tensor_result_a.dtype == np_result.dtype
    np.testing.assert_allclose(np_result, tensor_result_a, atol=1e-5, rtol=1e-5)

def test_scalar_methods():
    pass

def test_ewise_methods():
    pass
