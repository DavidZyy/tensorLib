// this file provides half precision matrix multiplication kernels using tensor core.
// reference:
// [1]: https://github.com/Bruce-Lee-LY/cuda_hgemm/tree/master

#include "device/CUDA.hpp"
#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include "util.hpp"
#include "device/ptx.hpp"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_SIZE 32

/************************************************************************************************************************************************************/
void hgemm_cublas(const half* lhs, const half* rhs, half* result, size_t M, size_t N, size_t K) {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        half alpha = 1.0f;
        half beta = 0.0f;

        CUBLAS_CHECK(cublasHgemm(
            handle, 
            CUBLAS_OP_T, CUBLAS_OP_N, 
            N, M, K, 
            &alpha, 
            rhs, K, // set to N will cause error 
            lhs, K, 
            &beta, 
            result, N));

        CUBLAS_CHECK(cublasDestroy(handle));
}

/************************************************************************************************************************************************************/
// naive implementation
// A is row major, B is column major
// only support aligned matrix now
// see https://zhuanlan.zhihu.com/p/650374808
__global__ void hgemm_kernel_v0(const half *A, const half *B, half *C, size_t M, size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ half A_smem[MMA_M][MMA_K];
    __shared__ half B_smem[MMA_N][MMA_K]; // B is column major. "__shared__ half B_smem[MMA_K][MMA_N];" is wrong.
    __shared__ half C_smem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RA[4];
    uint32_t RB[2];
    uint32_t RC[2] = {0, 0};

    #pragma unroll
    for (size_t i = 0; i < K_tiles; i++) {
        // load A and B from global memory to shared memory
        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) = *((int4 *)(A + (warp_row + lane_id / 2) * K + i * MMA_K) + lane_id % 2);

        if (lane_id < 16) {
            *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =  *((int4 *)(B + (warp_col + lane_id / 2) * K  + i * MMA_K) + lane_id % 2);
        }

        __syncthreads();

        // load from shared memory to registers
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);

        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        __syncthreads();
    }

// see https://pica.zhimg.com/v2-aef0a60b6976aeaeaa922151232d7b8a_r.jpg

// write from registers to shared memory
    *((int *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
    *((int *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

// write from shared memory to global memory

    // each thread write int2
    // *((int *)(C + (warp_row + lane_id / 2) * N + warp_col + (lane_id % 2 * 4))) = *((int *)(&C_smem[lane_id / 2][0] + lane_id % 2 * 4));
    // *((int *)(C + (warp_row + lane_id / 2) * N + warp_col + (lane_id % 2 * 4)) + 1) = *((int *)(&C_smem[lane_id / 2][0] + lane_id % 2 * 4) + 1); 

    // or the first 16 threads write int4
    if (lane_id < 16) {
        *((int4 *)(C + (warp_row + lane_id % 16) * N + warp_col)) = *((int4 *)(&C_smem[lane_id % 16][0]));
    }

}

void hgemm_v0(const half *A, const half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

    hgemm_kernel_v0<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

/************************************************************************************************************************************************************/
