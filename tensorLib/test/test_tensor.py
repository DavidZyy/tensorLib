import pytest
import tensor_bindings as tb  # the module is a .so file compiled from C++
import numpy as np
import random
import operator
import torch

def generate_random_shapes(n_shapes, min_dims=0, max_dims=4, max_size=10) -> list[tuple]:
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


def generate_batched_matmul_shapes(num_batch_dims_range=(1, 4), batch_size_range=(2, 4), dim_range=(100, 500)):
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
    # Randomly select the number of batch dimensions
    num_batch_dims = random.randint(*num_batch_dims_range)

    # Randomly generate batch sizes for either 1 or 2 batch dimensions
    batch_shape = tuple(random.randint(batch_size_range[0], batch_size_range[1]) for _ in range(num_batch_dims))

    # Randomly select dimensions for shape1 and shape2
    m = random.randint(dim_range[0], dim_range[1])  # Rows in shape1
    k = random.randint(dim_range[0], dim_range[1])  # Shared dimension (columns in shape1, rows in shape2)
    n = random.randint(dim_range[0], dim_range[1])  # Columns in shape2

    a = random.randint(0, num_batch_dims)
    b = random.randint(0, num_batch_dims)

    # Construct shape1 and shape2 with batch dimensions
    shape1 = tuple(list(batch_shape)[a:]) + (m, k)  # test batched matmul broadcasting
    shape2 = tuple(list(batch_shape)[b:]) + (k, n)  # test batched matmul broadcasting

    return shape1, shape2


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


bached_matmul_shapes = [generate_batched_matmul_shapes() for _ in range(50)]
@pytest.mark.parametrize("shape1, shape2", bached_matmul_shapes)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_matmul(shape1, shape2, device):
    A = np.random.randn(*shape1).astype(np.float32)  # must convert to float32!!! or it will be float64!!
    B = np.random.randn(*shape2).astype(np.float32)
    C = np.matmul(A, B)

    A_t = tb.convert_to_tensor(A, device)
    B_t = tb.convert_to_tensor(B, device)
    C_t = A_t.matmul(B_t)
    C_t_a = tb.convert_to_numpy(C_t)

    assert C.shape == C_t_a.shape
    assert C.dtype == C_t_a.dtype
    assert C.size == C_t_a.size
    np.testing.assert_allclose(C, C_t_a, atol=1e-4, rtol=1e-4)


reduced_ops = [
    ('sum', np.sum),
    ('mean', np.mean),
    ('max', np.max),
    ('min', np.min),
    ('argmax', np.argmax),
    ('argmin', np.argmin)
]
reduced_shapes = generate_random_shapes(50, min_dims=1, max_dims=4, max_size=100)
@pytest.mark.parametrize("shape", reduced_shapes)
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


# Test binary operators, like addition, subtraction, multiplication, division, and power
binary_shapes = generate_random_shapes(50, min_dims=1, max_dims=4, max_size=10)
binary_ops = [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]
@pytest.mark.parametrize("shape", binary_shapes)
@pytest.mark.parametrize("op", binary_ops)
@pytest.mark.parametrize("operand", ["scalar", "tensor"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_binary_methods(shape, op, operand, device):
    # do not support power operation with tensor operand
    if op == operator.pow and operand == "tensor":
        pytest.skip()

    A = np.random.randn(*shape).astype(np.float32)  # Random numpy array
    rhs = random.uniform(-10, 10) if operand == "scalar" else np.random.randn(*shape).astype(np.float32)
    # Apply the operation using numpy
    A_result = op(A, rhs)

    A_t = tb.convert_to_tensor(A, device)  # Convert to tensor
    rhs_t = rhs if operand == "scalar" else tb.convert_to_tensor(rhs, device)
    # rhs = tb.convert_to_tensor(rhs)  # get error in np.testing, seems like memory is broken

    # Apply the operation using the tensor's method
    if op == operator.pow:
        A_t_result = A_t.pow(rhs_t)  # For power operation, use `pow` method
    else:
        A_t_result = op(A_t, rhs_t)

    # Convert tensor result back to numpy for comparison
    A_t_result_a = tb.convert_to_numpy(A_t_result)

    # Compare results
    np.testing.assert_allclose(A_result, A_t_result_a, atol=1e-5, rtol=1e-5)



unary_ops = [
    ('neg', operator.neg),
    ('sin', np.sin),
    ('cos', np.cos),
    ('exp', np.exp),
    ('log', np.log),
    ('abs', np.abs),
    ('tanh', np.tanh),
    ('silu', lambda x: x * (1 / (1 + np.exp(-x)))),  # SILU function
    ('sqrt', np.sqrt),
    ('rsqrt', lambda x: 1 / np.sqrt(x))
]

# Use smaller shape sizes for simplicity
unary_shapes = generate_random_shapes(50, min_dims=1, max_dims=4, max_size=50)

@pytest.mark.parametrize("shape", unary_shapes)
@pytest.mark.parametrize("op_name, np_op", unary_ops)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
# @pytest.mark.parametrize("device", ["cpu"])
def test_unary_methods(shape, op_name, np_op, device):
    # Generate random data for the tensor
    A = np.random.randn(*shape).astype(np.float32)

    # Some operations like log, sqrt, rsqrt need positive values to avoid invalid operations
    if op_name in ['log', 'sqrt', 'rsqrt']:
        A = np.abs(A) + 1e-5  # Ensure positive values

    # Apply the numpy operation
    np_result = np_op(A)

    # Convert A to tensor and apply the tensor operation
    A_t = tb.convert_to_tensor(A, device)
    if op_name == "neg":
        tensor_result = np_op(A_t)
    else:
        tensor_op = getattr(A_t, op_name)  # Get the corresponding tensor operation
        tensor_result = tensor_op()

    # Convert tensor result back to numpy for comparison
    tensor_result_a = tb.convert_to_numpy(tensor_result)

    # Validate shapes and contents
    assert np_result.shape == tensor_result_a.shape
    assert np_result.dtype == tensor_result_a.dtype
    assert np_result.size == tensor_result_a.size
    np.testing.assert_allclose(np_result, tensor_result_a, atol=1e-5, rtol=1e-5)


def generate_random_tensor(shape, device) -> tuple[np.ndarray, tb.Tensor_fp32]:
    """Helper function to create a random numpy array and tensor with the same shape."""
    np_data = np.random.randn(*shape).astype(np.float32)
    tensor_data = tb.convert_to_tensor(np_data, device)
    return np_data, tensor_data


def generate_random_slices(shape) -> list[list[int]]:
    """Helper function to generate random slices for a given shape."""
    slices = []
    for dim in shape:
        start = random.randint(0, dim - 1)
        stop = random.randint(start + 1, dim)
        slices.append([start, stop, 1])
    return slices


getitem_shapes = generate_random_shapes(50, min_dims=1, max_dims=4, max_size=10)
@pytest.mark.parametrize("shape", getitem_shapes)
# @pytest.mark.parametrize("shape", [(10, 3)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
# @pytest.mark.parametrize("device", ["cpu"])
def test_getItem(shape, device):
    np_data, tensor_data = generate_random_tensor(shape, device)
    slices = generate_random_slices(shape)

    # np_data = np.ones((10, 3)).astype(np.float32)
    # tensor_data = tb.convert_to_tensor(np_data, device)
    # slices = [[6,8,1],[2,3,1]]

    # Convert slice format to Python slicing for numpy compatibility
    np_slices = tuple(slice(*slc) if slc else slice(None) for slc in slices)
    # np_slices = list(slice(*slc) if slc else slice(None) for slc in slices)

    # Apply the slices to numpy and tensor versions
    np_result = np_data[np_slices]
    # tensor_result = tensor_data.getItem(list(np_slices))
    tensor_result = tensor_data[np_slices]
    tensor_result_np = tb.convert_to_numpy(tensor_result)

    # Assert shapes and values
    assert np_result.shape == tensor_result_np.shape
    assert np_result.dtype == tensor_result_np.dtype
    assert np_result.size == tensor_result_np.size
    np.testing.assert_allclose(np_result, tensor_result_np, atol=1e-5, rtol=1e-5)


def compute_shape_after_slices(slices: list[list[int]]) -> tuple[int]:
    new_shape = []
    for start, stop, step in slices:
        new_shape.append((stop - start) // step)
    return tuple(new_shape)


setitem_shapes = generate_random_shapes(50, min_dims=1, max_dims=4, max_size=100)
@pytest.mark.parametrize("shape", setitem_shapes)
@pytest.mark.parametrize("operand", ["scalar", "tensor"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
# @pytest.mark.parametrize("operand", ["tensor"])
def test_setItem(shape, operand, device):
    np_data, _ = generate_random_tensor(shape, device)
    np_data_copy = np_data.copy()  # deep copy, so np_data and tensor_data are use the different memory
    tensor_data = tb.convert_to_tensor(np_data_copy, device)
    slices = generate_random_slices(shape)
    slice_shape = compute_shape_after_slices(slices)

    if operand == "scalar":
        np_set = random.uniform(-10, 10)
        tensor_set = np_set
    else:
        np_set, tensor_set = generate_random_tensor(slice_shape, device)

    np_slices = tuple(slice(*slc) if slc else slice(None) for slc in slices)

    np_data[np_slices] = np_set
    tensor_data[np_slices] = tensor_set
    tensor_data_np = tb.convert_to_numpy(tensor_data)

    assert np_data.shape == tensor_data_np.shape
    assert np_data.dtype == tensor_data_np.dtype
    assert np_data.size == tensor_data_np.size
    np.testing.assert_allclose(np_data, tensor_data_np, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_broadcast_to(device):
    inital_shape = (3, 1)
    target_shape = (3, 4)

    np_data, tensor_data = generate_random_tensor(inital_shape, device)
    np_result = np.broadcast_to(np_data, target_shape)
    tensor_result = tensor_data.broadcast_to(target_shape)
    tensor_result_np = tb.convert_to_numpy(tensor_result)
    assert np_result.shape == tensor_result_np.shape
    assert np_result.dtype == tensor_result_np.dtype
    assert np_result.size == tensor_result_np.size
    np.testing.assert_allclose(np_result, tensor_result_np, atol=1e-5, rtol=1e-5)

def test_reshape():
    pass

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_permute(device):
    initial_shape = (2, 3, 4)
    permuted_shape = (3, 4, 2)
    axes = (1, 2, 0)  # This is the permutation order

    np_data, tensor_data = generate_random_tensor(initial_shape, device)
    np_result = np.transpose(np_data, axes)  # Use NumPy to permute the dimensions
    tensor_result = tensor_data.permute(axes)  # Call your permute method
    tensor_result_np = tb.convert_to_numpy(tensor_result)  # Convert back to NumPy for comparison

    assert np_result.shape == tensor_result_np.shape
    assert np_result.dtype == tensor_result_np.dtype
    assert np_result.size == tensor_result_np.size
    np.testing.assert_allclose(np_result, tensor_result_np, atol=1e-5, rtol=1e-5)
