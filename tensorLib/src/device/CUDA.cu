#include "CUDA.hpp"
#include "Tensor.hpp"
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <library_types.h>
#include <stdexcept>
#include <vector>

template class CUDA<float>;
template class CUDA<int>;

template <typename dtype>
CUDA<dtype>::CUDA(size_t size) : Device<dtype>(size) {
    CUDA_CHECK(cudaMalloc(&this->data_, size * sizeof(dtype)));
}

template <typename dtype>
CUDA<dtype>::~CUDA() {
    CUDA_CHECK(cudaFree(this->data_));
}

/**
 * I think this implementation is not efficient, but it should be good enough for now.
 * We can improve it later.
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
void CUDA<dtype>::matmul(const dtype* lhs, const dtype* rhs, dtype* result, 
    const std::vector<int>& lhs_stride, 
    const std::vector<int>& rhs_stride, 
    size_t lhs_offset,
    size_t rhs_offset,
    const std::vector<int>& result_shape,
    size_t result_elements,
    size_t K)
{
    int ndim = result_shape.size();

    // Allocate device memory
    int* d_lhs_stride;
    int* d_rhs_stride;
    int* d_result_shape;
    CUDA_CHECK(cudaMalloc(&d_lhs_stride, ndim * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rhs_stride, ndim * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result_shape, ndim * sizeof(int)));

    // Copy strides and shapes to device memory
    CUDA_CHECK(cudaMemcpy(d_lhs_stride, lhs_stride.data(), ndim * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs_stride, rhs_stride.data(), ndim * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_result_shape, result_shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice));

    // Launch the kernel
    int threads_per_block = 256;
    int blocks = (result_elements + threads_per_block - 1) / threads_per_block;
    matmulKernel<<<blocks, threads_per_block>>>(lhs, rhs, result, 
                                                d_lhs_stride, d_rhs_stride, 
                                                lhs_offset, rhs_offset, 
                                                d_result_shape, result_elements, K, ndim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_lhs_stride));
    CUDA_CHECK(cudaFree(d_rhs_stride));
    CUDA_CHECK(cudaFree(d_result_shape));
}

template <typename dtype>
__global__ void fullKernel(dtype* data, size_t num_elements, dtype fill_value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = static_cast<dtype>(fill_value);
    }
}

template <typename dtype>
void CUDA<dtype>::full(size_t num_elements, dtype fill_value) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    fullKernel<<<blocks_per_grid, threads_per_block>>>(this->data_, num_elements, fill_value);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename dtype>
dtype CUDA<dtype>:: getDataLinear(size_t linear_index) const {
    dtype result;
    CUDA_CHECK(cudaMemcpy(&result, this->data_ + linear_index, sizeof(dtype), cudaMemcpyDeviceToHost));
    return result;
}

template <typename dtype>
__global__ void contiguous_kernel(
    dtype* result,
    const dtype* data,
    const int* shape,
    const int* stride,
    size_t offset,
    size_t num_elements,
    int dim_size) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        size_t linear_index_new = 0;
        size_t linear_index = i;

        for (int j = dim_size - 1; j >= 0; --j) {
            int cur_dim_id = linear_index % shape[j];
            linear_index /= shape[j];
            linear_index_new += cur_dim_id * stride[j];
        }

        result[i] = data[linear_index_new + offset];
    }
}

template <typename dtype>
void CUDA<dtype>::contiguous(
    dtype* result,
    const std::vector<int>& shape,
    const std::vector<int>& stride,
    size_t offset,
    size_t num_elements) 
{
    // Allocate memory for shape and stride on the device
    int* d_shape;
    int* d_stride;
    cudaMalloc(&d_shape, shape.size() * sizeof(int));
    cudaMalloc(&d_stride, stride.size() * sizeof(int));

    // Copy shape and stride data to device
    cudaMemcpy(d_shape, shape.data(), shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stride, stride.data(), stride.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    contiguous_kernel<<<num_blocks, threads_per_block>>>(
        result, this->data_, d_shape, d_stride, offset, num_elements, shape.size());

    // Free device memory for shape and stride
    cudaFree(d_shape);
    cudaFree(d_stride);
}
