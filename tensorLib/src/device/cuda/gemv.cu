// reference: 
// [1] git@github.com:Bruce-Lee-LY/cuda_hgemv.git
// [2] git@github.com:wangsiping97/FastGEMV.git
// in decode stage of transformer, the operation is gemv, which is a matrix-vector multiplication
#include "device/CUDA.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "util.hpp"

template class CUDA<int8_t>;
template class CUDA<half>;
template class CUDA<float>;
template class CUDA<int>;

#define WARP_SIZE 32

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
// threads
#define WARPS_PER_BLOCK_v0 4
#define THREADS_PER_BLOCK_v0  (WARP_SIZE * WARPS_PER_BLOCK_v0)

template<typename dtype>
__global__ void gemv_kernel_v0(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
    const size_t col = blockIdx.x * THREADS_PER_BLOCK_v0 + threadIdx.x;
    if (col >= N) return;

    float tmp = 0.0;

    #pragma unroll
    for (size_t i = 0; i < K; ++i) {
        tmp += static_cast<float>(A[i]) * static_cast<float>(B[i + col * K]); // B is col major
    }

    C[col] = static_cast<dtype>(tmp);
}

template<typename dtype>
void gemv_v0(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
    dim3 block(THREADS_PER_BLOCK_v0);
    dim3 grid(div_ceil(N, THREADS_PER_BLOCK_v0));
    gemv_kernel_v0<<<grid, block>>>(A, B, C, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void gemv_v0<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t N, size_t K);
template void gemv_v0<half>(const half* A, const half* B, half* C, size_t N, size_t K);
template void gemv_v0<float>(const float* A, const float* B, float* C, size_t N, size_t K);
template void gemv_v0<int>(const int* A, const int* B, int* C, size_t N, size_t K);

/************************************************************************************************************************************************************/
// // threads + shared memory
// #define WARP_SIZE 32
// #define WARPS_PER_BLOCK 4
// #define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK
// template<typename dtype>
// __global__ void gemv_kernel_v1(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
//     // printf("k: %d\n", K);
//     extern __shared__ float A_smem[];
//     // __shared__ float A_smem[2048];
//     // assert(K <= 2048);
// 
//     size_t A_smem_iters = div_ceil(K, THREADS_PER_BLOCK);
// 
//     // fetch A from global memory to shared memory
//     # pragma unroll
//     for (size_t i = 0; i < A_smem_iters; ++i) {
//         size_t idx = i * THREADS_PER_BLOCK + threadIdx.x;
//         if (idx < K) {
//             A_smem[idx] = A[idx];
//         }
//     }
// 
//     __syncthreads();
// 
//     const size_t col = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
//     if (col >= N) return;
// 
//     float tmp = 0.0;
//     # pragma unroll
//     for (size_t i = 0; i < K; ++i) {
//         tmp += A[i] * B[i + col * K]; // B is col major
//         // tmp += A_smem[i] * B[i + col * K]; // B is col major
//         // tmp += A_smem[i] * B[i*N + col];  // B is row major
//         // tmp += B[i*N + col];  // B is row major
//     }
// 
//     C[col] = tmp;
// }
// 
// template <typename dtype>
// void gemv_v1(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
// //     static size_t smem_size = initThreadSmem(K); call this every time when call gemv, very slow!!!
// 
//     dim3 block(THREADS_PER_BLOCK);
//     dim3 grid(div_ceil(N, THREADS_PER_BLOCK));
//     // gemvKernelV1<<<grid, block>>>(A, B, C, M, N, K);
//     gemv_kernel_v1<<<grid, block, K * sizeof(dtype)>>>(A, B, C, N, K);
//     // gemvKernelV1<<<grid, block, 1024*2>>>(A, B, C, M, N, K);
//     // gemvKernelV1<<<grid, block, smem_size>>>(A, B, C, M, N, K);
//     // static size_t smm = 4*256;
//     // gemvKernelV1<<<grid, block, smm>>>(A, B, C, M, N, K);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());
// }
// 
// template void gemv_v1<float>(const float* A, const float* B, float* C, size_t N, size_t K);
// template void gemv_v1<int>(const int* A, const int* B, int* C, size_t N, size_t K);
// template void gemv_v1<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t N, size_t K);

/************************************************************************************************************************************************************/
// #define WARP_SIZE 32
// #define WARPS_PER_BLOCK 4
// #define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK
// template <typename dtype>
// __global__ void gemv_kernel_v2(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
//     const size_t warp_id = threadIdx.x / WARP_SIZE; // the warp index which this thread belongs to
//     const size_t warp_col = blockIdx.x * WARPS_PER_BLOCK + warp_id; // the column index this warp will process
// 
//     if (warp_col >= N) return;
// 
//     const size_t K_iters = div_ceil(K, WARP_SIZE); // the number of iterations to process all elements in A
//     const size_t lane_id = threadIdx.x % WARP_SIZE; // the lane index in the warp
// 
//     dtype tmp = 0.0;
// #pragma unroll
//     for (size_t i = 0; i < K_iters; ++i) {
//         size_t A_idx = i * WARP_SIZE + lane_id;
//         size_t B_idx = i * WARP_SIZE + lane_id + warp_col * K;
//         if (A_idx < K) {
//             tmp += A[A_idx] * B[B_idx];
//         }
//     }
// 
//     constexpr unsigned int mask = 0xffffffff;
// #pragma unroll
//     for (size_t i = WARP_SIZE / 2; i >= 1; i /= 2) {
//         tmp += __shfl_xor_sync(mask, tmp, i);
//     }
// 
//     if (lane_id == 0) {
//         C[warp_col] = tmp;
//     }
// }
// 
// template<typename dtype>
// void gemv_v2(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
//     dim3 block(THREADS_PER_BLOCK);
//     dim3 grid(div_ceil(N, WARPS_PER_BLOCK));
//     gemv_kernel_v2<<<grid, block>>>(A, B, C, N, K);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());
// }
// 
// template void gemv_v2<float>(const float* A, const float* B, float* C, size_t N, size_t K);
// template void gemv_v2<int>(const int* A, const int* B, int* C, size_t N, size_t K);
// template void gemv_v2<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t N, size_t K);

/************************************************************************************************************************************************************/
#define WARPS_PER_BLOCK_v3 4
#define THREADS_PER_BLOCK_v3  (WARP_SIZE * WARPS_PER_BLOCK_v3)

template <typename dtype>
__global__ void gemv_kernel_v3(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
//     // if constexpr (std::is_same_v<dtype, half>) {
//         // __shared__ dtype A_smem[4096 * 4];
//     // } else {
//         __shared__ dtype A_smem[4096 * 2];
//     // }
//     size_t A_smem_iters = div_ceil(K, THREADS_PER_BLOCK_v3);
// 
// #pragma unroll
//     for (size_t i = 0; i < A_smem_iters; ++i) {
//         size_t idx = i * THREADS_PER_BLOCK_v3 + threadIdx.x;
//         if (idx < K) {
//             A_smem[idx] = A[idx];
//         }
//     }
// 
//     __syncthreads();

    const size_t warp_id = threadIdx.x / WARP_SIZE; // the warp index which this thread belongs to
    const size_t warp_col = blockIdx.x * WARPS_PER_BLOCK_v3 + warp_id; // the column index this warp will process

    if (warp_col >= N) return;

    const size_t K_iters = div_ceil(K, WARP_SIZE); // the number of iterations to process all elements in A
    const size_t lane_id = threadIdx.x % WARP_SIZE; // the lane index in the warp

    float tmp = 0.0;
#pragma unroll
    for (size_t i = 0; i < K_iters; ++i) {
        size_t A_idx = i * WARP_SIZE + lane_id;
        size_t B_idx = i * WARP_SIZE + lane_id + warp_col * K;
        if (A_idx < K) {
            // tmp += static_cast<float>(A_smem[A_idx]) * static_cast<float>(B[B_idx]);
            tmp += static_cast<float>(A[A_idx]) * static_cast<float>(B[B_idx]);
        }
    }

    const unsigned int mask = 0xffffffff;

#pragma unroll
    for (size_t i = WARP_SIZE / 2; i >= 1; i /= 2) {
        tmp += __shfl_xor_sync(mask, tmp, i);
    }

    if (lane_id == 0) {
        C[warp_col] = static_cast<dtype>(tmp);
    }
}

template<typename dtype>
void gemv_v3(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
    dim3 block(THREADS_PER_BLOCK_v3);
    dim3 grid(div_ceil(N, WARPS_PER_BLOCK_v3));
    gemv_kernel_v3<<<grid, block>>>(A, B, C, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void gemv_v3<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t N, size_t K);
template void gemv_v3<half>(const half* A, const half* B, half* C, size_t N, size_t K);
template void gemv_v3<float>(const float* A, const float* B, float* C, size_t N, size_t K);
template void gemv_v3<int>(const int* A, const int* B, int* C, size_t N, size_t K);

/************************************************************************************************************************************************************/
#define WARPS_PER_BLOCK_v4 4
#define THREADS_PER_BLOCK_v4 (WARP_SIZE * WARPS_PER_BLOCK_v4)

#define COLS_PER_WARP_v4 1 // each warp processes columns
#define COLS_PER_BLOCK_v4  (COLS_PER_WARP_v4 * WARPS_PER_BLOCK_v4)
#define THREADS_PER_GROUP_v4 (WARP_SIZE / COLS_PER_WARP_v4)

template <typename dtype>
__global__ void gemv_kernel_v4(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
    const size_t group_id = threadIdx.x / THREADS_PER_GROUP_v4;
    const size_t group_col = blockIdx.x * COLS_PER_BLOCK_v4 + group_id;

    if (group_col >= N) return;

    const size_t K_iters = div_ceil(K, THREADS_PER_GROUP_v4);
    const size_t group_lane_id = threadIdx.x % THREADS_PER_GROUP_v4;

    float tmp = 0.0f;
#pragma unroll
    for (size_t i = 0; i < K_iters; ++i) {
        size_t A_idx = i * THREADS_PER_GROUP_v4 + group_lane_id;
        size_t B_idx = i * THREADS_PER_GROUP_v4 + group_lane_id + group_col * K;
        if (A_idx < K) {
            tmp += static_cast<float>(A[A_idx]) * static_cast<float>(B[B_idx]);
        }
    }

    const unsigned int mask = 0xffffffff;
#pragma unroll
    for (size_t i = THREADS_PER_GROUP_v4 / 2; i >= 1; i /= 2) {
        tmp += __shfl_xor_sync(mask, tmp, i);
    }

    if (group_lane_id == 0) {
        C[group_col] = static_cast<dtype>(tmp);
    }
}

template<typename dtype>
void gemv_v4(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
    // initThreadSmem(K);
    dim3 block(THREADS_PER_BLOCK_v4);
    dim3 grid(div_ceil(N, COLS_PER_BLOCK_v4));
    // gemv_kernel_v3<<<grid, block, K*sizeof(dtype)>>>(A, B, C, N, K);
    gemv_kernel_v4<<<grid, block>>>(A, B, C, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


template void gemv_v4<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t N, size_t K);
template void gemv_v4<half>(const half* A, const half* B, half* C, size_t N, size_t K);
template void gemv_v4<float>(const float* A, const float* B, float* C, size_t N, size_t K);
template void gemv_v4<int>(const int* A, const int* B, int* C, size_t N, size_t K);

/************************************************************************************************************************************************************/
/**
 * MAKE SURE the address of each row is aligned to 16 bytes.
 * For example, if B is 4097 * 4097, the first row of B may be aligned to 16 bytes, 
 * but if the second row is consecutive to the first row in memory, it not be aligned to 16 bytes.
 */
#define WARPS_PER_BLOCK_v5 4
#define THREADS_PER_BLOCK_v5 (WARP_SIZE * WARPS_PER_BLOCK_v5)
template<typename dtype>
__global__ void gemv_kernel_v5(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
    if constexpr (std::is_same_v<dtype, half>) {
        // raise not support half error
        assert(0);
    } else { // dtype = float
        const size_t col = blockIdx.x * THREADS_PER_BLOCK_v5 + threadIdx.x;
        if (col >= N) return;

        float4 tmp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (size_t i = 0; i < K; i+=4) {
            if (i + 4 <= K) {
                float4 a = *(reinterpret_cast<const float4*>(A + i));
                float4 b = *(reinterpret_cast<const float4*>(B + col * K + i));
                tmp.x += a.x * b.x;
                tmp.y += a.y * b.y;
                tmp.z += a.z * b.z;
                tmp.w += a.w * b.w;
            } else {
                for (size_t j = i; j < K; ++j) {
                    tmp.x += A[j] * B[j + col * K];
                }
            }
        }

        C[col] = tmp.x + tmp.y + tmp.z + tmp.w;
    }
}

template<typename dtype>
void gemv_v5(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
    assert (K % 4 == 0); // else will raise misaligned address error.
    dim3 block(THREADS_PER_BLOCK_v5);
    dim3 grid(div_ceil(N, THREADS_PER_BLOCK_v5));
    gemv_kernel_v5<<<grid, block>>>(A, B, C, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void gemv_v5<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t N, size_t K);
template void gemv_v5<half>(const half* A, const half* B, half* C, size_t N, size_t K);
template void gemv_v5<float>(const float* A, const float* B, float* C, size_t N, size_t K);
template void gemv_v5<int>(const int* A, const int* B, int* C, size_t N, size_t K);

/************************************************************************************************************************************************************/
__device__ __forceinline__ float warpReduceSum(float sum,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}

/************************************************************************************************************************************************************/
// reference: git@github.com:wangsiping97/FastGEMV.git
template<typename dtype>
__global__ void gemv_kernel_v6(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
if constexpr (std::is_same_v<dtype, half>) {

    const size_t tid = threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int i = tid * 8; i < K; i += 8 * blockDim.x) {
        float4 a = *(reinterpret_cast<const float4*>(A + i));
        float4 b = *(reinterpret_cast<const float4*>(B + col * K + i));
        // convert each float in float4 to half2
        half2 a0 = *(reinterpret_cast<half2*>(&a.x));
        half2 a1 = *(reinterpret_cast<half2*>(&a.y));
        half2 a2 = *(reinterpret_cast<half2*>(&a.z));
        half2 a3 = *(reinterpret_cast<half2*>(&a.w));
        half2 b0 = *(reinterpret_cast<half2*>(&b.x));
        half2 b1 = *(reinterpret_cast<half2*>(&b.y));
        half2 b2 = *(reinterpret_cast<half2*>(&b.z));
        half2 b3 = *(reinterpret_cast<half2*>(&b.w));
        sum += __half2float(a0.x) * __half2float(b0.x);
        sum += __half2float(a0.y) * __half2float(b0.y);
        sum += __half2float(a1.x) * __half2float(b1.x);
        sum += __half2float(a1.y) * __half2float(b1.y);
        sum += __half2float(a2.x) * __half2float(b2.x);
        sum += __half2float(a2.y) * __half2float(b2.y);
        sum += __half2float(a3.x) * __half2float(b3.x);
        sum += __half2float(a3.y) * __half2float(b3.y);
    }

    sum = warpReduceSum(sum, blockDim.x);

    if (blockDim.x <= WARP_SIZE) {
        if (tid == 0) {
          C[col] = __float2half(sum);
        }
    } else {
        assert(0);
    }

//     const unsigned int mask = 0xffffffff;
//     #pragma unroll
//     for (size_t i = blockDim.x / 2; i >= 1; i /= 2) {
//         sum += __shfl_xor_sync(mask, sum, i);
//     }
// 
//     if (tid == 0) {
//         C[col] = static_cast<dtype>(sum);
//     }

} else {
    assert(0);
}

}

template<typename dtype>
void gemv_v6(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
    assert (K % 4 == 0); // else will raise misaligned address error.

    int block_dim_x = 32; // number of threads handle a single row
    int block_dim_y = 4; // number of rows handled by a single block

    dim3 block(block_dim_x, block_dim_y);
    dim3 grid(1, div_ceil(N, block_dim_y));

    gemv_kernel_v6<<<grid, block>>>(A, B, C, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void gemv_v6<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t N, size_t K);
template void gemv_v6<half>(const half* A, const half* B, half* C, size_t N, size_t K);
template void gemv_v6<float>(const float* A, const float* B, float* C, size_t N, size_t K);
template void gemv_v6<int>(const int* A, const int* B, int* C, size_t N, size_t K);

/************************************************************************************************************************************************************/

// slower than v6 ...
template<typename dtype>
__global__ void gemv_kernel_v7(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
if constexpr (std::is_same_v<dtype, half>) {
    /////////////////////////////// load to shared memory ////////////////////////////////
    __shared__ dtype A_smem[4096 * 2];

    const size_t threads_per_block = blockDim.x * blockDim.y;
    size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    #pragma unroll
    for (size_t i = thread_id; i < K; i += threads_per_block) {
        A_smem[i] = A[i];
    }

    __syncthreads();
    /////////////////////////////// compute ////////////////////////////////
    const size_t tid = threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int i = tid * 8; i < K; i += 8 * blockDim.x) {
        float4 a = *(reinterpret_cast<const float4*>(A_smem + i));
        float4 b = *(reinterpret_cast<const float4*>(B + col * K + i));
        // convert each float in float4 to half2
        half2 a0 = *(reinterpret_cast<half2*>(&a.x));
        half2 a1 = *(reinterpret_cast<half2*>(&a.y));
        half2 a2 = *(reinterpret_cast<half2*>(&a.z));
        half2 a3 = *(reinterpret_cast<half2*>(&a.w));
        half2 b0 = *(reinterpret_cast<half2*>(&b.x));
        half2 b1 = *(reinterpret_cast<half2*>(&b.y));
        half2 b2 = *(reinterpret_cast<half2*>(&b.z));
        half2 b3 = *(reinterpret_cast<half2*>(&b.w));
        sum += __half2float(a0.x) * __half2float(b0.x);
        sum += __half2float(a0.y) * __half2float(b0.y);
        sum += __half2float(a1.x) * __half2float(b1.x);
        sum += __half2float(a1.y) * __half2float(b1.y);
        sum += __half2float(a2.x) * __half2float(b2.x);
        sum += __half2float(a2.y) * __half2float(b2.y);
        sum += __half2float(a3.x) * __half2float(b3.x);
        sum += __half2float(a3.y) * __half2float(b3.y);
    }

    sum = warpReduceSum(sum, blockDim.x);

    if (blockDim.x <= WARP_SIZE) {
        if (tid == 0) {
          C[col] = __float2half(sum);
        }
    } else {
        assert(0);
    }

//     const unsigned int mask = 0xffffffff;
//     #pragma unroll
//     for (size_t i = blockDim.x / 2; i >= 1; i /= 2) {
//         sum += __shfl_xor_sync(mask, sum, i);
//     }
// 
//     if (tid == 0) {
//         C[col] = static_cast<dtype>(sum);
//     }

}
}

template<typename dtype>
void gemv_v7(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
    assert (K % 4 == 0); // else will raise misaligned address error.

    int block_dim_x = 32; // number of threads handle a single row
    int block_dim_y = 16; // number of rows handled by a single block

    dim3 block(block_dim_x, block_dim_y);
    dim3 grid(1, div_ceil(N, block_dim_y));

    gemv_kernel_v7<<<grid, block>>>(A, B, C, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void gemv_v7<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t N, size_t K);
template void gemv_v7<half>(const half* A, const half* B, half* C, size_t N, size_t K);
template void gemv_v7<float>(const float* A, const float* B, float* C, size_t N, size_t K);
template void gemv_v7<int>(const int* A, const int* B, int* C, size_t N, size_t K);

/************************************************************************************************************************************************************/

template<typename dtype>
__global__ void gemv_kernel_v8(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
if constexpr (std::is_same_v<dtype, half>) {

    const size_t tid = threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int i = tid * 8; i < K; i += 8 * blockDim.x) {
        float4 a = *(reinterpret_cast<const float4*>(A + i));
        float4 b = *(reinterpret_cast<const float4*>(B + col * K + i));
        // convert each float in float4 to half2
        half2 a0 = *(reinterpret_cast<half2*>(&a.x));
        half2 a1 = *(reinterpret_cast<half2*>(&a.y));
        half2 a2 = *(reinterpret_cast<half2*>(&a.z));
        half2 a3 = *(reinterpret_cast<half2*>(&a.w));
        half2 b0 = *(reinterpret_cast<half2*>(&b.x));
        half2 b1 = *(reinterpret_cast<half2*>(&b.y));
        half2 b2 = *(reinterpret_cast<half2*>(&b.z));
        half2 b3 = *(reinterpret_cast<half2*>(&b.w));
        sum += __half2float(a0.x) * __half2float(b0.x);
        sum += __half2float(a0.y) * __half2float(b0.y);
        sum += __half2float(a1.x) * __half2float(b1.x);
        sum += __half2float(a1.y) * __half2float(b1.y);
        sum += __half2float(a2.x) * __half2float(b2.x);
        sum += __half2float(a2.y) * __half2float(b2.y);
        sum += __half2float(a3.x) * __half2float(b3.x);
        sum += __half2float(a3.y) * __half2float(b3.y);
    }

    sum = warpReduceSum(sum, blockDim.x);

    if (blockDim.x <= WARP_SIZE) {
        if (tid == 0) {
          C[col] = __float2half(sum);
        }
        return;
    }


    assert(blockDim.y <= 32); // row num
    assert(blockDim.x <= WARP_SIZE * WARP_SIZE);

    __shared__ float warpLevelSums[32][WARP_SIZE];

    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    if (laneId == 0) {
        warpLevelSums[threadIdx.y][warpId] = sum;
    }

    __syncthreads();

    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        sum = warpLevelSums[threadIdx.y][laneId];
    } else {
        sum = 0.0f;
    }

    if (warpId == 0) {
        sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
    }

    if (tid == 0) {
        C[col] = __float2half(sum);
    }


} else {
    assert(0);
}

}

template<typename dtype>
void gemv_v8(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
    assert (K % 4 == 0); // else will raise misaligned address error.

    int block_dim_x = 32; // number of threads handle a single row
    int block_dim_y = 4; // number of rows handled by a single block

    dim3 block(block_dim_x, block_dim_y);
    dim3 grid(1, div_ceil(N, block_dim_y));

    gemv_kernel_v8<<<grid, block>>>(A, B, C, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void gemv_v8<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t N, size_t K);
template void gemv_v8<half>(const half* A, const half* B, half* C, size_t N, size_t K);
template void gemv_v8<float>(const float* A, const float* B, float* C, size_t N, size_t K);
template void gemv_v8<int>(const int* A, const int* B, int* C, size_t N, size_t K);

/************************************************************************************************************************************************************/

// copy from reference 2, not quicker than other.
// __global__ void gemv_fp16(const half* mat, const half* vec, half* res, unsigned int n,
//                           unsigned int num_per_thread) {
//   float sum = 0;
//   // each thread load num_per_thread elements from global
//   unsigned int tid = threadIdx.x;
//   unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
//   unsigned int start_idx = threadIdx.x;
//   const float4* mat4 = reinterpret_cast<const float4*>(mat);
//   const float4* vec4 = reinterpret_cast<const float4*>(vec);
// 
// #pragma unroll
//   for (int iter = 0; iter < num_per_thread >> 3; iter++) {
//     unsigned int j = start_idx + iter * blockDim.x;
//     if (j < n >> 3) {
//       float4 vec_val = vec4[j];
//       float4 mat_val = mat4[row * (n >> 3) + j];
//       const half2* vec_h1 = (half2*)&vec_val.x;
//       const half2* vec_h2 = (half2*)&vec_val.y;
//       const half2* vec_h3 = (half2*)&vec_val.z;
//       const half2* vec_h4 = (half2*)&vec_val.w;
//       const half2* mat_h1 = (half2*)&mat_val.x;
//       const half2* mat_h2 = (half2*)&mat_val.y;
//       const half2* mat_h3 = (half2*)&mat_val.z;
//       const half2* mat_h4 = (half2*)&mat_val.w;
//       sum += static_cast<float>(vec_h1->x) * static_cast<float>(mat_h1->x);
//       sum += static_cast<float>(vec_h1->y) * static_cast<float>(mat_h1->y);
//       sum += static_cast<float>(vec_h2->x) * static_cast<float>(mat_h2->x);
//       sum += static_cast<float>(vec_h2->y) * static_cast<float>(mat_h2->y);
//       sum += static_cast<float>(vec_h3->x) * static_cast<float>(mat_h3->x);
//       sum += static_cast<float>(vec_h3->y) * static_cast<float>(mat_h3->y);
//       sum += static_cast<float>(vec_h4->x) * static_cast<float>(mat_h4->x);
//       sum += static_cast<float>(vec_h4->y) * static_cast<float>(mat_h4->y);
//     }
//   }
// 
//   sum = warpReduceSum(sum, blockDim.x);
// 
//   if (blockDim.x <= WARP_SIZE) {
//     if (tid == 0) {
//       res[row] = __float2half(sum);
//     }
//     return;
//   }
// 
//   // Shared mem for partial sums (one per warp in the block)
//   static __shared__ float warpLevelSums[32][WARP_SIZE];
//   const int laneId = threadIdx.x % WARP_SIZE;
//   const int warpId = threadIdx.x / WARP_SIZE;
//   if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
//   __syncthreads();
//   // read from shared memory only if that warp existed
//   sum = (threadIdx.x < blockDim.x / WARP_SIZE)
//             ? warpLevelSums[threadIdx.y][laneId]
//             : 0.0;
//   // Final reduce using first warp
//   if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
//   if (tid == 0) {
//     res[row] = __float2half(sum);
//   }
// }
// 
// template<typename dtype>
// void gemv_v9(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
//     assert (K % 4 == 0); // else will raise misaligned address error.
// 
//     int block_dim_x = 32 * 4; // number of threads handle a single row
//     int block_dim_y = 4; // number of rows handled by a single block
// 
//     dim3 block(block_dim_x, block_dim_y);
//     dim3 grid(1, div_ceil(N, block_dim_y));
// 
//     gemv_fp16<<<grid, block>>>(B, A,  C, N, K / block_dim_x);
// 
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());
// }
// 
// template void gemv_v9<half>(const half* A, const half* B, half* C, size_t N, size_t K);

/************************************************************************************************************************************************************/

template<typename dtype>
void gemv_cublasSgemv(const dtype* A, const dtype* B, dtype* C, size_t N, size_t K) {
if constexpr (std::is_same<dtype, float>::value) {

    // Create a handle for cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Define scalars alpha and beta
    dtype alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasSgemv(
        handle,               // cuBLAS handle
        CUBLAS_OP_T,          // No transpose on Matrix
        K,                    // Number of rows in A
        N,                    // Number of columns in A
        &alpha,               // Scalar multiplier for A * B
        B, K,                 // Matrix  (size N x K)
        A, 1,                 // Vector  (size K)
        &beta,                // Scalar multiplier for C
        C, 1                  // Resulting vector C (size N)
    ));

    // Clean up cuBLAS handle
    cublasDestroy(handle);

} else {
    throw std::runtime_error("gemv_cublasSgemv only supports float.");
}

}

template void gemv_cublasSgemv<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, size_t N, size_t K);
template void gemv_cublasSgemv<half>(const half* A, const half* B, half* C, size_t N, size_t K);
template void gemv_cublasSgemv<float>(const float* A, const float* B, float* C, size_t N, size_t K);
template void gemv_cublasSgemv<int>(const int* A, const int* B, int* C, size_t N, size_t K);
/************************************************************************************************************************************************************/
