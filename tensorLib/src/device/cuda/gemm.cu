/**
 * @file gemm.cu
 * @author Yangyang Zhu (1929772352@qq.com)
 * @version 0.1
 * @date 2025-02-17
 * 
 * @copyright Copyright (c) 2025
 * this file provides gemm kernels which support dtype (both half and float)
 */
#include "device/cuda/CUDA.cuh"
#include <iostream>

template class CUDA<int8_t>;
template class CUDA<half>;
template class CUDA<float>;
template class CUDA<int>;

/**
 * I think this implementation is not efficient, but it should be good enough for now.
 * We can improve it later.
 * Maybe we can make lhs and rhs be contiguous, and use shared memory in one block ?
 * execuate contiguous in cpu is not efficient, so i give up this step when perform batched matmul in cpu,
 * but in cuda, maybe we can do it ?
 *
 * @tparam dtype 
 */
// batched matmul
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

// batched matmul
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

/************************************************************************************************************************************************************/
/**
 * 2D matrix multiplication
 * naive implementation
 * lhs is row major, rhs is col major
 * @tparam dtype 
 */
template <typename dtype>
__global__ void gemm_kernel_v0(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;  // Row index
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;  // Column index

    if (i < M && j < N) {

        float sum = 0.0f;

        #pragma unroll

        for (size_t k = 0; k < K; ++k) {
            if constexpr (std::is_same<dtype, half>::value) {
                sum += __half2float(lhs[i * K + k]) * __half2float(rhs[j * K + k]);
            } else {
                sum += lhs[i * K + k] * rhs[j * K + k];
            }
        }

        if constexpr (std::is_same<dtype, half>::value) {
            result[i * N + j] = __float2half(sum);
        } else {
            result[i * N + j] = sum;
        }
    }
}

template<typename dtype>
void gemm_v0(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K) {
    dim3 threadsPerBlock(16, 16);  // Define block size (16x16 is a typical choice, can be adjusted)
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);  // Number of blocks

    gemm_kernel_v0<<<numBlocks, threadsPerBlock>>>(lhs, rhs, result, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

template void gemm_v0<half>(const half* A, const half* B, half* C, size_t m, size_t n, size_t k);
template void gemm_v0<float>(const float* A, const float* B, float* C, size_t m, size_t n, size_t k);
/************************************************************************************************************************************************************/
// gemv
template<typename dtype> void gemv_v0(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K);
template<typename dtype> void gemv_v1(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K);
template<typename dtype> void gemv_v2(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K);
template<typename dtype> void gemv_v3(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K);
template<typename dtype> void gemv_v4(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K);
template<typename dtype> void gemv_v5(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K);
template<typename dtype> void gemv_v6(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K);
template<typename dtype> void gemv_cublasSgemv(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K);

// gemm
void hgemm_cublas(const half* lhs, const half* rhs, half* result, size_t M, size_t N, size_t K);
void hgemm_v0(const half* lhs, const half* rhs, half* result, size_t M, size_t N, size_t K);

/************************************************************************************************************************************************************/
/**
 * NOTE: lhs is row major, rhs is col major !!!
 */
template<typename dtype>
void CUDA<dtype>::matmul2d(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K) {
    if (M == 1) {
        // use gemv kernel
        if (K == 11008 || K == 4096) {
            // lhs and rhs align to 16 bytes, use gemv_6 kernel
            gemv_v6(lhs, rhs, result, N, K);
        } else {
            gemv_v4(lhs, rhs, result, N, K);
        }
    } else {
        // gemm_v0(lhs, rhs, result, M, N, K);

        // if dtype is half
        if constexpr (std::is_same<dtype, half>::value) {
            hgemm_cublas(lhs, rhs, result, M, N, K);
            // hgemm_v0(lhs, rhs, result, M, N, K);
        } 
    }
}
