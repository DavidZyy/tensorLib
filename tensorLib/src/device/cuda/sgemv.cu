// reference: git@github.com:Bruce-Lee-LY/cuda_hgemv.git
// in decode stage of transformer, the operation is gemv, which is a matrix-vector multiplication
#include "device/CUDA.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "util.hpp"

template class CUDA<float>;
template class CUDA<int>;
template class CUDA<int8_t>;

/************************************************************************************************************************************************************/
size_t initThreadSmem(size_t K) {
    int dev_id = 0;
    CUDA_CHECK(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size = K * sizeof(float);
    // assert(dev_prop.sharedMemPerMultiprocessor >= smem_max_size);
    assert(dev_prop.sharedMemPerBlock >= smem_max_size);
    // printf("Shared memory per multiprocessor: %zu\n", dev_prop.sharedMemPerMultiprocessor);
    // printf("Max shared memory per block: %zu\n", dev_prop.sharedMemPerBlock);
    // printf("smem_max_size: %zu\n", smem_max_size);

    // CUDA_CHECK(cudaFuncSetAttribute(gemvKernelV1<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

/************************************************************************************************************************************************************/
// // threads
// #define WARP_SIZE 32
// #define WARPS_PER_BLOCK 4
// #define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK
// template<typename dtype>
// __global__ void gemvKernelV0(const dtype* A, const dtype* B, dtype* C, size_t M, size_t N, size_t K) {
//     const size_t col = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
//     if (col >= N) return;
// 
//     float tmp = 0.0;
// 
//     #pragma unroll
//     for (size_t i = 0; i < K; ++i) {
//         tmp += A[i] * B[i + col * K]; // B is col major
//         // tmp += A[i] * B[i*N + col];  // B is row major
//     }
// 
//     C[col] = tmp;
// }
// 
// template<typename dtype>
// void gemvV0(const dtype* A, const dtype* B, dtype* C, size_t M, size_t N, size_t K) {
//     dim3 block(THREADS_PER_BLOCK);
//     dim3 grid(div_ceil(N, THREADS_PER_BLOCK));
//     gemvKernelV0<<<grid, block>>>(A, B, C, M, N, K);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());
// }
// 
// template void gemvV0<float>(const float* A, const float* B, float* C, size_t M, size_t N, size_t K);
// template void gemvV0<int>(const int* A, const int* B, int* C, size_t M, size_t N, size_t K);
// template void gemvV0<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t M, size_t N, size_t K);

/************************************************************************************************************************************************************/
// threads + shared memory
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK
template<typename dtype>
__global__ void gemvKernelV1(const dtype* A, const dtype* B, dtype* C, size_t M, size_t N, size_t K) {
    // printf("k: %d\n", K);
    extern __shared__ float A_smem[];
    // __shared__ float A_smem[2048];
    // assert(K <= 2048);

    size_t A_smem_iters = div_ceil(K, THREADS_PER_BLOCK);

    // fetch A from global memory to shared memory
    # pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        size_t idx = i * THREADS_PER_BLOCK + threadIdx.x;
        if (idx < K) {
            A_smem[idx] = A[idx];
        }
    }

    __syncthreads();

    const size_t col = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (col >= N) return;

    float tmp = 0.0;
    # pragma unroll
    for (size_t i = 0; i < K; ++i) {
        tmp += A[i] * B[i + col * K]; // B is col major
        // tmp += A_smem[i] * B[i + col * K]; // B is col major
        // tmp += A_smem[i] * B[i*N + col];  // B is row major
        // tmp += B[i*N + col];  // B is row major
    }

    C[col] = tmp;
}

template <typename dtype>
void gemvV1(const dtype* A, const dtype* B, dtype* C, size_t M, size_t N, size_t K) {
    static size_t smem_size = initThreadSmem(K);

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, THREADS_PER_BLOCK));
    // gemvKernelV1<<<grid, block>>>(A, B, C, M, N, K);
    gemvKernelV1<<<grid, block, K * sizeof(dtype)>>>(A, B, C, M, N, K);
    // gemvKernelV1<<<grid, block, 1024*2>>>(A, B, C, M, N, K);
    // gemvKernelV1<<<grid, block, smem_size>>>(A, B, C, M, N, K);
    // static size_t smm = 4*256;
    // gemvKernelV1<<<grid, block, smm>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void gemvV1<float>(const float* A, const float* B, float* C, size_t M, size_t N, size_t K);
template void gemvV1<int>(const int* A, const int* B, int* C, size_t M, size_t N, size_t K);
template void gemvV1<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t M, size_t N, size_t K);

/************************************************************************************************************************************************************/
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK
template <typename dtype>
__global__ void gemv_kernel_v2(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
    const size_t warp_id = threadIdx.x / WARP_SIZE; // the warp index which this thread belongs to
    const size_t warp_col = blockIdx.x * WARPS_PER_BLOCK + warp_id; // the column index this warp will process

    if (warp_col >= N) return;

    const size_t K_iters = div_ceil(K, WARP_SIZE); // the number of iterations to process all elements in A
    const size_t lane_id = threadIdx.x % WARP_SIZE; // the lane index in the warp

    dtype tmp = 0.0;
#pragma unroll
    for (size_t i = 0; i < K_iters; ++i) {
        size_t A_idx = i * WARP_SIZE + lane_id;
        size_t B_idx = i * WARP_SIZE + lane_id + warp_col * K;
        if (A_idx < K) {
            tmp += A[A_idx] * B[B_idx];
        }
    }

    constexpr unsigned int mask = 0xffffffff;
#pragma unroll
    for (size_t i = WARP_SIZE / 2; i >= 1; i /= 2) {
        tmp += __shfl_xor_sync(mask, tmp, i);
    }

    if (lane_id == 0) {
        C[warp_col] = tmp;
    }
}

template<typename dtype>
void gemv_v2(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, WARPS_PER_BLOCK));
    gemv_kernel_v2<<<grid, block>>>(A, B, C, N, K);
}

template void gemv_v2<float>(const float* A, const float* B, float* C, size_t N, size_t K);
template void gemv_v2<int>(const int* A, const int* B, int* C, size_t N, size_t K);
template void gemv_v2<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t N, size_t K);

/************************************************************************************************************************************************************/
