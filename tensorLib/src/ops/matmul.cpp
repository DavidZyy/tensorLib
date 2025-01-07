#include "ops/matmul.hpp"
#include <cstdint>
#include <cstdio>
#include <utility>

namespace ops {
template struct matmul<float>;
template struct matmul<int>;
template struct matmul<int8_t>;

/**
 * should use ops::methods to replace Tensor.methods in the following codes.
 * split bached gemm to singel gemms...
 * @tparam dtype 
 */

template <typename dtype>
Tensor<dtype> matmul<dtype>::call(const Tensor<dtype> &self, const Tensor<dtype> &other) {
    if (self.device_type != other.device_type) {
        throw std::invalid_argument("Tensors must be on the same device.");
    }

    // Ensure dimensionality is compatible for matrix multiplication
    if (self.ndim < 2 || other.ndim < 2) {
        throw std::invalid_argument("Tensors must have at least 2 dimensions for matmul.");
    }

    // The last dimension of A should match the second-to-last dimension of B
    if (self.shape_[self.ndim - 1] != other.shape_[other.ndim - 2]) {
        throw std::invalid_argument("Shape mismatch: the number of columns in the first tensor must match the number of rows in the second tensor.");
    }

    Tensor<dtype> A = self;
    Tensor<dtype> B = other;

    size_t num_batch_dims = std::max(A.ndim - 2, B.ndim - 2);
    size_t dim_diff = std::abs(static_cast<int>(A.ndim) - static_cast<int>(B.ndim));

    // If needed, prepend dimensions to match larger tensor size
    std::vector<int> A_broadcast_shape = A.shape_;
    std::vector<int> B_broadcast_shape = B.shape_;
    std::vector<int> output_shape;

    // for example, A.shape = (2, 2, 3, 4, 5), B.shape = (3, 5, 4), after this, B.shape will be (1, 1, 3, 5, 4) -> (2, 2, 3, 5, 4)
    if (A.ndim < B.ndim) {
        A_broadcast_shape.insert(A_broadcast_shape.begin(), dim_diff, 1);
    } else if (B.ndim < A.ndim) {
        B_broadcast_shape.insert(B_broadcast_shape.begin(), dim_diff, 1);
    }

    // Adjust batch dimensions to be broadcast-compatible
    for (size_t i = 0; i < num_batch_dims; ++i) {
        if (A_broadcast_shape[i] != B_broadcast_shape[i]) {
            if (A_broadcast_shape[i] == 1) {
                A_broadcast_shape[i] = B_broadcast_shape[i];
            } else if (B_broadcast_shape[i] == 1) {
                B_broadcast_shape[i] = A_broadcast_shape[i];
            } else {
                throw std::invalid_argument("Shape mismatch: the batch dimensions must be broadcastable.");
            }
        }
        output_shape.push_back(A_broadcast_shape[i]);
    }

    A = A.broadcast_to(A_broadcast_shape);
    B = B.broadcast_to(B_broadcast_shape);

    output_shape.push_back(A.shape_[A.ndim - 2]);
    output_shape.push_back(B.shape_[B.ndim - 1]);
    int height = A.shape_[A.ndim - 2];
    int width = B.shape_[B.ndim - 1];
    int K = A.shape_[A.ndim - 1];

    // now execute batched matmul
    Tensor<dtype> result(output_shape, self.device_type); // forget pass in device_type will get bug !!

    size_t result_elements = result.num_elements;

    self.device->matmul(
        self.device->getDataPtr(),
        other.device->getDataPtr(),
        result.device->getDataPtr(),
        A.stride_,
        B.stride_,
        A.offset_,
        B.offset_,
        result.shape_,
        result_elements,
        K);

    return result;
}

/**
 * call matmul2D
 * @tparam dtype 
 */
template <typename dtype>
Tensor<dtype> matmul<dtype>::call2(const Tensor<dtype> &self, const Tensor<dtype> &other) {
    if (self.device_type != other.device_type) {
        throw std::invalid_argument("Tensors must be on the same device.");
    }

    // Ensure dimensionality is compatible for matrix multiplication
    if (self.ndim < 2 || other.ndim < 2) {
        throw std::invalid_argument("Tensors must have at least 2 dimensions for matmul.");
    }

    // The last dimension of A should match the second-to-last dimension of B
    if (self.shape_[self.ndim - 1] != other.shape_[other.ndim - 2]) {
        throw std::invalid_argument("Shape mismatch: the number of columns in the first tensor must match the number of rows in the second tensor.");
    }

    Tensor<dtype> A = self;
    Tensor<dtype> B = other;

    // make it contiguous for matmul
    A = A.contiguous();
    B = B.contiguous();

    size_t num_batch_dims = std::max(A.ndim - 2, B.ndim - 2);
    size_t dim_diff = std::abs(static_cast<int>(A.ndim) - static_cast<int>(B.ndim));

    // If needed, prepend dimensions to match larger tensor size
    std::vector<int> A_broadcast_shape = A.shape_;
    std::vector<int> B_broadcast_shape = B.shape_;
    std::vector<int> output_shape;

    // for example, A.shape = (2, 2, 3, 4, 5), B.shape = (3, 5, 4), after this, B.shape will be (1, 1, 3, 5, 4) -> (2, 2, 3, 5, 4)
    if (A.ndim < B.ndim) {
        A_broadcast_shape.insert(A_broadcast_shape.begin(), dim_diff, 1);
    } else if (B.ndim < A.ndim) {
        B_broadcast_shape.insert(B_broadcast_shape.begin(), dim_diff, 1);
    }

    // Adjust batch dimensions to be broadcast-compatible
    for (size_t i = 0; i < num_batch_dims; ++i) {
        if (A_broadcast_shape[i] != B_broadcast_shape[i]) {
            if (A_broadcast_shape[i] == 1) {
                A_broadcast_shape[i] = B_broadcast_shape[i];
            } else if (B_broadcast_shape[i] == 1) {
                B_broadcast_shape[i] = A_broadcast_shape[i];
            } else {
                throw std::invalid_argument("Shape mismatch: the batch dimensions must be broadcastable.");
            }
        }
        output_shape.push_back(A_broadcast_shape[i]);
    }

    A = A.broadcast_to(A_broadcast_shape);
    B = B.broadcast_to(B_broadcast_shape);

    output_shape.push_back(A.shape_[A.ndim - 2]);
    output_shape.push_back(B.shape_[B.ndim - 1]);

    int M = A.shape_[A.ndim - 2];
    int N = B.shape_[B.ndim - 1];
    int K = A.shape_[A.ndim - 1];

    // now execute batched matmul
    Tensor<dtype> C(output_shape, self.device_type); // forget pass in device_type will get bug !!

    // split batched gemm to single gemms
    // getItem of A and B and C, which will hold the same memory with them, and then do gemm 
    int matrix_num = 1;
    for (int i = 0; i < num_batch_dims; ++i) {
        matrix_num *= A_broadcast_shape[i];
    }

    for (int i = 0; i < matrix_num; ++i) {
        std::vector<std::vector<int>> slices;

        int num = i;
        for (int j = 0; j < num_batch_dims; j++) {
            int cur_id = num % A_broadcast_shape[j];
            slices.push_back({cur_id});
            num = num / A_broadcast_shape[j];
        }
        slices.push_back({});
        slices.push_back({});

        Tensor<dtype> A_item = A.getItem(slices);
        Tensor<dtype> B_item = B.getItem(slices);
        Tensor<dtype> C_item = C.getItem(slices);

        self.device->matmul2d(
            A_item.device->getDataPtr() + A_item.offset_,
            B_item.device->getDataPtr() + B_item.offset_,
            C_item.device->getDataPtr() + C_item.offset_,
            M,
            N,
            K
        );
    }
    
    return C;
}

} // namespace ops
