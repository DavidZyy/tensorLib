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
                             CudaVec lhs_stride, CudaVec rhs_stride, 
                             size_t lhs_offset, size_t rhs_offset,
                             CudaVec result_shape, size_t result_elements,
                             size_t K) 
{
    size_t ndim = result_shape.size;
    // Global thread index for each result element
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= result_elements) return;

    size_t linear_index = idx;
    size_t Aoff = lhs_offset, Boff = rhs_offset;

    // Compute offsets for lhs and rhs
    for (int i = ndim - 1; i >= 0; --i) {
        int cur_dim_id = linear_index % result_shape.data[i];
        linear_index /= result_shape.data[i];

        if (i != ndim - 1)
            Aoff += cur_dim_id * lhs_stride.data[i];
        if (i != ndim - 2)
            Boff += cur_dim_id * rhs_stride.data[i];
    }

    // Compute the dot product
    dtype sum = 0;
    int t1 = lhs_stride.data[ndim - 1], t2 = rhs_stride.data[ndim - 2];
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
    // Launch the kernel
    int threads_per_block = 256;
    int blocks = (result_elements + threads_per_block - 1) / threads_per_block;
    matmulKernel<<<blocks, threads_per_block>>>(lhs, rhs, result, 
                                                VecToCuda(lhs_stride), VecToCuda(rhs_stride), 
                                                lhs_offset, rhs_offset, 
                                                VecToCuda(result_shape), result_elements, K);
    CUDA_CHECK(cudaGetLastError());
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
    CudaVec shape,
    CudaVec stride,
    size_t offset,
    size_t num_elements) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        size_t linear_index_new = convertIdx(i, shape, stride, offset);
        
        result[i] = data[linear_index_new];
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
    // Calculate grid and block dimensions
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    contiguous_kernel<<<num_blocks, threads_per_block>>>(
        result, this->data_, VecToCuda(shape), VecToCuda(stride), offset, num_elements);
    CUDA_CHECK(cudaGetLastError());
}

template <typename dtype>
__global__ void setItemEwiseKernel(
    dtype* data,
    const dtype* src,
    CudaVec shape,
    CudaVec stride, 
    size_t offset,
    size_t num_elements) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        size_t linearIdx = convertIdx(i, shape, stride, offset);
        data[linearIdx] = src[i];
    }
}

template <typename dtype>
void CUDA<dtype>::setItemEwise(
    dtype* src,
    const std::vector<int>& shape,
    const std::vector<int>& stride,
    size_t offset,
    size_t num_elements) 
{
    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (num_elements + blockSize - 1) / blockSize;

    // Launch the kernel
    setItemEwiseKernel<<<gridSize, blockSize>>>(
        this->data_, src, VecToCuda(shape), VecToCuda(stride), offset, num_elements);

    CUDA_CHECK(cudaGetLastError());
}

template <typename dtype>
__global__ void setItemScalarKernel(
    dtype* data,
    const dtype value,
    CudaVec shape,
    CudaVec stride, 
    size_t offset,
    size_t num_elements) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        size_t linearIdx = convertIdx(i, shape, stride, offset);
        data[linearIdx] = value;
    }
}

template <typename dtype>
void CUDA<dtype>::setItemScalar(
    dtype value,
    const std::vector<int>& shape,
    const std::vector<int>& stride,
    size_t offset,
    size_t num_elements) 
{
    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (num_elements + blockSize - 1) / blockSize;

    // Launch the kernel
    setItemScalarKernel<<<gridSize, blockSize>>>(
        this->data_, value, VecToCuda(shape), VecToCuda(stride), offset, num_elements);

    CUDA_CHECK(cudaGetLastError());
}
