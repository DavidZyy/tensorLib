// reference: https://zhuanlan.zhihu.com/p/341059988
// https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/softmax.cuh#L223
#include "device/cuda/CUDA.cuh"
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <type_traits>
#include "util.hpp"
#include "device/cuda/warp.cuh"

template class CUDA<int8_t>;
template class CUDA<half>;
template class CUDA<float>;
template class CUDA<int>;

#define WARP_SIZE 32

// suppose the memory layout of input is a 2D matrix. A non-2d matrix is converted to 2d in Tensor layer.
// use a warp handle a row
template <typename dtype>
__global__ void softmaxKernel_v0(dtype* output, const dtype* input, size_t cols) {
// if (blockIdx.x == 0) {

    const size_t row = blockIdx.x * cols;

    dtype thread_max = -INFINITY;
    __shared__ dtype warp_max;

    dtype thread_sum = 0.0;
    __shared__ dtype warp_sum;

    if (threadIdx.x == 0) {
        warp_max = -INFINITY;
        warp_sum = 0.0;
    }

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if constexpr (std::is_same_v<dtype, float>) {
            thread_max = max(thread_max, input[row + i]);
        } else {
            thread_max = static_cast<dtype>(max(static_cast<float>(thread_max), static_cast<float>(input[row + i])));
        }
    }

    warp_max = warpReduceMax<dtype>(thread_max); 
    // printf thread idx, thread_max, warp_max
    // printf("thread_idx: %d, thread_max: %f, warp_max: %f\n", threadIdx.x, thread_max, warp_max);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if constexpr (std::is_same_v<dtype, float>) {
            output[row + i] = exp(input[row + i] - warp_max);
        } else {
            output[row + i] = static_cast<dtype>(exp(static_cast<float>(input[row + i]) - static_cast<float>(warp_max)));
        }

        thread_sum += output[row + i];
    }

    warp_sum = warpReduceSum<dtype>(thread_sum);
    // printf("thread_idx: %d, thread_sum: %f, warp_sum: %f\n", threadIdx.x, thread_sum, warp_sum);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        output[row + i] /= warp_sum;
    }

}

// one warp in a block handle a row
template<typename dtype>
void softmax_v0(dtype* output, const dtype* input, size_t rows, size_t cols) {
    dim3 block(WARP_SIZE);
    dim3 grid(rows);

    softmaxKernel_v0<dtype><<<grid, block>>>(output, input, cols);
    CUDA_CHECK(cudaGetLastError());
}

template<typename dtype>
void CUDA<dtype>::softmax(dtype* output, size_t rows, size_t cols) const {
    softmax_v0<dtype>(output, this->data_, rows, cols);
}
