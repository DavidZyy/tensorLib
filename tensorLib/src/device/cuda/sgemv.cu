// reference: git@github.com:Bruce-Lee-LY/cuda_hgemv.git
// in decode stage of transformer, the operation is gemv, which is a matrix-vector multiplication
#include "device/CUDA.hpp"
#include <cstdint>
#include <iostream>
#include "util.hpp"

template class CUDA<float>;
template class CUDA<int>;
template class CUDA<int8_t>;


#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

// threads
template<typename dtype>
__global__ void gemvKernelV0(const dtype* A, const dtype* B, dtype* C, size_t M, size_t N, size_t K) {
    const size_t col = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (col >= N) return;

    float tmp = 0.0;

    #pragma unroll
    for (size_t i = 0; i < K; ++i) {
        // tmp += A[i] * B[i + col * K]; // B is col major
        tmp += A[i] * B[i*N + col];  // B is row major
    }

    C[col] = tmp;
}

template<typename dtype>
void gemvV0(const dtype* A, const dtype* B, dtype* C, size_t M, size_t N, size_t K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, THREADS_PER_BLOCK));
    gemvKernelV0<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void gemvV0<float>(const float* A, const float* B, float* C, size_t M, size_t N, size_t K);
template void gemvV0<int>(const int* A, const int* B, int* C, size_t M, size_t N, size_t K);
template void gemvV0<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t M, size_t N, size_t K);
