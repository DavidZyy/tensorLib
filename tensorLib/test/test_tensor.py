import pytest
import tensor_bindings as tb
import numpy as np
import random
# import libtensor_bindings as tb
# from . import libtensor_bindings as tb


def generate_random_shapes(n_shapes, max_dims=4, max_size=10):
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
        num_dims = random.randint(0, max_dims)
        if num_dims == 0:
            shapes.append(())  # Scalar shape
        else:
            # Generate a shape with random sizes for each dimension
            shape = tuple(random.randint(1, max_size) for _ in range(num_dims))
            shapes.append(shape)
    
    return shapes


convert_shapes = generate_random_shapes(100, max_dims=4, max_size=100)


@pytest.mark.parametrize("shape", convert_shapes)
def test_convert(shape):
    print(shape)
    # Generate random data
    A = np.random.randn(*shape)

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
