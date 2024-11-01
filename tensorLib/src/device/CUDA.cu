#include "CUDA.hpp"
#include "Tensor.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

template <typename dtype>
CUDA<dtype>::CUDA(size_t num_elements) {
    cudaError_t err = cudaMalloc(&this->data_, num_elements * sizeof(dtype));
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    }
}

template <typename dtype>
CUDA<dtype>::~CUDA() {
    cudaFree(this->data_);
}

/**
 * I think this implementation is not efficient, but it should be good enough for now.
 * We can always improve it later.
 * Maybe we can make lhs and rhs be contiguous, and use shared memory in one block ?
 * execuate contiguous in cpu is not efficient, so i give up this step when perform batched matmul in cpu,
 * but in cuda, maybe we can do it ?
 *
 * @tparam dtype 
 */
template <typename dtype>
__global__ void matmulKernel(const dtype* lhs, const dtype* rhs, dtype* result, 
                             const int* lhs_stride, const int* rhs_stride, 
                             size_t lhs_offset, size_t rhs_offset,
                             const int* result_shape, size_t result_elements,
                             size_t K,
                             size_t ndim) {
    // Global thread index for each result element
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= result_elements) return;

    size_t linear_index = idx;
    size_t Aoff = lhs_offset, Boff = rhs_offset;
    int row, col;

    // Compute offsets for lhs and rhs
    for (int i = ndim - 1; i >= 0; --i) {
        int cur_dim_id = linear_index % result_shape[i];
        linear_index /= result_shape[i];

        if (i != ndim - 1)
            Aoff += cur_dim_id * lhs_stride[i];
        if (i != ndim - 2)
            Boff += cur_dim_id * rhs_stride[i];
    }

    // Compute the dot product
    dtype sum = 0;
    int t1 = lhs_stride[ndim - 1], t2 = rhs_stride[ndim - 2];
    for (int k = 0; k < K; ++k) {
        sum += lhs[Aoff + k * t1] * rhs[Boff + k * t2];
    }

    // Store the result
    result[idx] = sum;
}

// Wrapper function to launch the CUDA kernel
template <typename dtype>
void CUDA<dtype>::matmul(dtype* lhs, dtype* rhs, dtype* result, 
                        std::vector<int>& lhs_stride, 
                        std::vector<int>& rhs_stride, 
                        size_t lhs_offset,
                        size_t rhs_offset,
                        std::vector<int>& result_shape, 
                        size_t result_elements,
                        size_t K) {

    int ndim = result_shape.size();

    // Allocate device memory
    int* d_lhs_stride;
    int* d_rhs_stride;
    int* d_result_shape;
    cudaMalloc(&d_lhs_stride, ndim * sizeof(int));
    cudaMalloc(&d_rhs_stride, ndim * sizeof(int));
    cudaMalloc(&d_result_shape, ndim * sizeof(int));

    // Copy strides and shapes to device memory
    cudaMemcpy(d_lhs_stride, lhs_stride.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhs_stride, rhs_stride.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_shape, result_shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threads_per_block = 256;
    int blocks = (result_elements + threads_per_block - 1) / threads_per_block;
    matmulKernel<<<blocks, threads_per_block>>>(lhs, rhs, result, 
                                                d_lhs_stride, d_rhs_stride, 
                                                lhs_offset, rhs_offset, 
                                                d_result_shape, result_elements, K, ndim);

    // Free device memory
    cudaFree(d_lhs_stride);
    cudaFree(d_rhs_stride);
    cudaFree(d_result_shape);
}
