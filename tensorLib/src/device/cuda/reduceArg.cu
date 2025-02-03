#include "device/CUDA.hpp"
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include "util.hpp"

template class CUDA<int8_t>;
template class CUDA<half>;
template class CUDA<float>;
template class CUDA<int>;

template <typename dtype> 
static inline __device__ bool argmaxFunc(dtype a, dtype b) { 
    return a > b; 
}
template <typename dtype> 
static inline __device__ bool argminFunc(dtype a, dtype b) { 
    return a < b; 
}

/************************************************************************************************************************************************************/
/**
 * use a thread to reduce a row
 * @tparam dtype 
 * @tparam (*comp)(dtype, dtype) 
 */
template <typename dtype, bool (*comp)(dtype, dtype)>
__global__ void reduceArgKernel_v0(int* result, const dtype* data, size_t reduce_size, size_t num_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements / reduce_size) {
        dtype best_value = data[i * reduce_size];
        int best_idx = 0;
        for (int j = 1; j < reduce_size; j++) {
            if (comp(data[i * reduce_size + j], best_value)) {
                best_value = data[i * reduce_size + j];
                best_idx = j;
            }
        }
        result[i] = best_idx;
    }
}

template <typename dtype, bool (*comp)(dtype, dtype)>
void reduceArg_v0(int* result, const dtype* data, size_t reduce_size, size_t num_elements) {
    int blockSize = 256;  // Number of threads per block (adjust based on optimization needs)
    // int gridSize = (num_elements / reduce_size + blockSize - 1) / blockSize;  // Number of blocks
    int gridSize = div_ceil(num_elements / reduce_size, blockSize);  // Number of blocks

    reduceArgKernel_v0<dtype, comp><<<gridSize, blockSize>>>(result, data, reduce_size, num_elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/************************************************************************************************************************************************************/
#define THREADS_PER_BLOCK 64
/**
 * use a block to reduce a row
 */
template <typename dtype, bool (*comp)(dtype, dtype)>
__global__ void reduceArgKernel_v1(int* result, const dtype* data, size_t reduce_size, size_t num_elements) {
    // const dtype* reduce_row = data + blockIdx.x * reduce_size; // the row to reduce
    const dtype* reduce_row = data;

    dtype best_value;
    int best_idx;

    best_value = reduce_row[threadIdx.x];
    best_idx = threadIdx.x;
    // printf("best_idx: %d\n", best_idx);

    // if (threadIdx.x < reduce_size) {
    //     best_value = reduce_row[threadIdx.x];
    //     best_idx = threadIdx.x;
    // } else {
    //     best_idx = -1; // can not be 
    // }
    __shared__ dtype shared_best_value[THREADS_PER_BLOCK];
    __shared__ dtype shared_best_idx[THREADS_PER_BLOCK];

                // printf("threadIdx.x: %d, best_value: %d, best_idx: %d\n", threadIdx.x, (int)best_value, best_idx);
    for (size_t i = THREADS_PER_BLOCK; i < reduce_size; i += THREADS_PER_BLOCK) {
        int idx = i + threadIdx.x;
        if (idx < reduce_size) {
            if (comp(reduce_row[idx], best_value)) {
                best_value = reduce_row[idx];
                best_idx = idx;
            }
        }
    }

    shared_best_value[threadIdx.x] = best_value; // forget this,  get bug!!
    shared_best_idx[threadIdx.x] = best_idx;

    // reduce whole block
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
        // printf("stride: %d\n", stride);
        __syncthreads();
        if (threadIdx.x < stride) {
            // if (comp(reduce_row[threadIdx.x + stride], best_value)) { // error !!!!
            // if (comp(shared_best_value[threadIdx.x + stride], best_value) { // error !!!
            if (comp(shared_best_value[threadIdx.x + stride], shared_best_value[threadIdx.x])) {
                // best_value = reduce_row[threadIdx.x + stride];
                // best_idx = threadIdx.x + stride;
                // best_value = shared_best_value[threadIdx.x + stride];
                // best_idx = shared_best_idx[threadIdx.x + stride];
                // printf("threadIdx.x: %d, best_value: %d, best_idx: %d\n", threadIdx.x, (int)best_value, best_idx);
                shared_best_value[threadIdx.x] = shared_best_value[threadIdx.x + stride];
                shared_best_idx[threadIdx.x] = shared_best_idx[threadIdx.x + stride];
            }
        }
    }

    // write result
    if (threadIdx.x == 0) {
        // printf("blockIdx.x: %d, best_idx: %d\n", blockIdx.x, best_idx);
        result[blockIdx.x] = shared_best_idx[threadIdx.x];
    }
}

/**rows = 1 in llm last step of get the max prob*/
template <typename dtype, bool (*comp)(dtype, dtype)>
void reduceArg_v1(int* result, const dtype* data, size_t reduce_size, size_t num_elements) {
    assert(reduce_size >= THREADS_PER_BLOCK);
    size_t rows = num_elements / reduce_size;
    // int blockSize = 256;  // Number of threads per block (adjust based on optimization needs)
    // int gridSize = (num_elements / reduce_size + blockSize - 1) / blockSize;  // Number of blocks
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(rows);

    reduceArgKernel_v1<dtype, comp><<<grid, block>>>(result, data, reduce_size, num_elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/************************************************************************************************************************************************************/
template <typename dtype>
template <bool (*comp)(dtype, dtype)>
void CUDA<dtype>::reduceOperationArg(int* result, size_t reduce_size, size_t num_elements) const {
    // reduceArg_v0<dtype, comp>(result, this->data_, reduce_size, num_elements);
    reduceArg_v1<dtype, comp>(result, this->data_, reduce_size, num_elements);
    // if (reduce_size >= THREADS_PER_BLOCK) {
    //     reduceArg_v1<dtype, comp>(result, this->data_, reduce_size, num_elements);
    // } else {
    //     reduceArg_v0<dtype, comp>(result, this->data_, reduce_size, num_elements);
    // }
}

/************************************************************************************************************************************************************/
template <typename dtype> 
void CUDA<dtype>::argmax(int* result, size_t reduce_size, size_t num_elements) const { 
    reduceOperationArg<argmaxFunc<dtype>>(result, reduce_size, num_elements); 
}

template <typename dtype> 
void CUDA<dtype>::argmin(int* result, size_t reduce_size, size_t num_elements) const { 
    reduceOperationArg<argminFunc<dtype>>(result, reduce_size, num_elements); 
}

// template <typename dtype, bool (*comp)(dtype, dtype)>
// __global__ void reduceArgKernel(int* result, const dtype* data, size_t reduce_size, size_t num_elements) {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < num_elements / reduce_size) {
//         dtype best_value = data[i * reduce_size];
//         int best_idx = 0;
//         for (int j = 1; j < reduce_size; j++) {
//             if (comp(data[i * reduce_size + j], best_value)) {
//                 best_value = data[i * reduce_size + j];
//                 best_idx = j;
//             }
//         }
//         result[i] = best_idx;
//     }
// }
// 
// template <typename dtype>
// template <bool (*comp)(dtype, dtype)>
// void CUDA<dtype>::reduceOperationArg(int* result, size_t reduce_size, size_t num_elements) const {
//     int blockSize = 256;  // Number of threads per block (adjust based on optimization needs)
//     int gridSize = (num_elements / reduce_size + blockSize - 1) / blockSize;  // Number of blocks
// 
//     reduceArgKernel<dtype, comp><<<gridSize, blockSize>>>(result, this->data_, reduce_size, num_elements);
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());
// }
