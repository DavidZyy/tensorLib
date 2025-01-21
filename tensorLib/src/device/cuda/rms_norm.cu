#include "device/CUDA.hpp"
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <library_types.h>
#include <iostream>
#include <vector>
#include "util.hpp"

template class CUDA<float>;
// template class CUDA<int>;
// template class CUDA<int8_t>;
// Explicit template instantiation for different types
// template void CUDA<float>::rms_norm<float>(float* output, float* input, float* weight, float epsilon, int hidden_size, int num_tokens);
// template void CUDA<int>::rms_norm<int>(int* output, int* input, int* weight, float epsilon, int hidden_size, int num_tokens);
// template<> void CUDA<float>::rms_norm(float* output, float* input, float* weight, float epsilon, int hidden_size, int num_tokens);
// template<> void CUDA<int>::rms_norm(int* output, int* input, int* weight, float epsilon, int hidden_size, int num_tokens);

/*********************************************************************************************************************/
#define HandleNum 1 // the number of elements that each thread handles
/** the precision error is too big ( > 1e-3) compared to pytorch version */
template<typename dtype>
__global__ void rms_norm_kernel_v0(dtype *output, dtype *input, dtype *weight, float epsilon, int hidden_size) {
    const int bidx = blockIdx.x;
    const int tidx = threadIdx.x;

    // printf("bidx: %d, tidx: %d\n", bidx, tidx);

    // extern __shared__ dtype shared_mem[];
    // dtype* input_mem = shared_mem;
    // dtype* input2_mem = shared_mem + hidden_size; // used for parallel reduction
    // dtype* weight_mem = shared_mem + 2 * hidden_size;
    // __shared__ dtype rms;

    extern __shared__ dtype shared_mem[];
    // dtype input_mem[hidden_size];
    dtype* input2_mem = shared_mem; // used for parallel reduction
    // dtype weight_mem[hidden_size];
    __shared__ dtype rms;

    // fetch input and weight into shared memory
    // maybe have bank conflict here!!! because 1 thread access continuous address in shared memory
    for (int i = tidx * HandleNum; i < (tidx + 1) * HandleNum; i++) {
        if (i < hidden_size) {
            // input_mem[i]  = input[bidx * hidden_size + i];
            dtype val = input[bidx * hidden_size + i];
            // printf("input[%d]: %f\n", i, input_mem[i]);
            // input2_mem[i] = input_mem[i] * input_mem[i];
            input2_mem[i] = val * val;
            // weight_mem[i] = weight[i];
        }
    }

    // parallel reduction to calculate the sum of input * input
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tidx < stride) {
            for (int i = 0; i < HandleNum; i++) {
                input2_mem[tidx * HandleNum + i] += input2_mem[(tidx + stride) * HandleNum + i];
            }
        }   
    }

    // calculate rms
    if (tidx == 0) {
        rms = 0.0; // forget to initialize rms cause the precision error!!!!!!!!!!!!!!!!!!!
        for (int i = 0; i < HandleNum; i++) {
            rms += input2_mem[i];
        }
        rms = sqrtf(rms / hidden_size + epsilon);
        // rms = sqrt(rms / hidden_size + epsilon);
        // dtype rms_r = rsqrtf()
    }

    __syncthreads();

    // calculate output
    for (int i = tidx * HandleNum; i < (tidx + 1) * HandleNum; i++) {
        if (i < hidden_size) {
            // dtype val = input_mem[i];
            dtype val = input[bidx * hidden_size + i];
            dtype w = weight[i];
            output[bidx * hidden_size + i] = val * w / rms;
            // printf("bidx: %d, tid: %d, val: %f, w: %f, rms: %f\n", bidx, tidx, val, w, rms);
            // printf("output[%d]: %f\n", i, output[bidx * hidden_size + i]);
        }
    }
}

/**
 * a usual shape is (batch_size, sequence_length, hidden_size)
 * fused rms norm kernel
 * @tparam dtype 
 * num_tokens: number of tokens, batch size * sequence length
 */
template<typename dtype>
void rms_norm_v0(dtype *output, dtype *input, dtype *weight, float epsilon, int hidden_size, int num_tokens) {
    assert(hidden_size % HandleNum == 0);
    int gridSize = num_tokens;
    int blockSize = hidden_size / HandleNum;

    // assert(blockSize <= 1024);
    // assert(gridSize <= 1024);

    // get the max block size and grid size
    int blockSizeLimit;
    cudaDeviceGetAttribute(&blockSizeLimit, cudaDevAttrMaxThreadsPerBlock, 0);
    // std::cout << "Block Size Limit: " << blockSizeLimit << std::endl;
    assert(blockSize <= blockSizeLimit);

    int gridSizeLimit;
    cudaDeviceGetAttribute(&gridSizeLimit, cudaDevAttrMaxGridDimX, 0);
    // std::cout << "Grid Size Limit: " << gridSizeLimit << std::endl;
    assert(gridSize <= gridSizeLimit);


    int shared_mem_size = hidden_size * sizeof(dtype);
    int smem;
    cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    // std::cout << "Shared Memory Limit: " << smem << " bytes" << std::endl;
    assert(shared_mem_size + sizeof(dtype) <= smem); // 1 more dtype for rms
    
    rms_norm_kernel_v0<<<gridSize, blockSize, shared_mem_size>>>(output, input, weight, epsilon, hidden_size);

    CUDA_CHECK(cudaGetLastError()); // if shared memory is not enough or grid / block size too large, it will return an error here.
    CUDA_CHECK(cudaDeviceSynchronize());
}

/*********************************************************************************************************************/
// reference: https://github.com/karpathy/llm.c
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// version 1 of rms norm kernel, one warp handles one token row
template <typename dtype>
__global__ void rms_norm_kernel_v1(dtype *output, dtype *input, dtype *weight, float epsilon, int hidden_size, int num_tokens) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // the warp index, the row(token) index
    if (idx >= num_tokens) 
        return;

    // the row of input that this group of threads will process
    const dtype *x = input + idx * hidden_size;

    // mean
    dtype sum = 0.0f; // aussume dtype is float
    for (int i = warp.thread_rank(); i < hidden_size; i += warp.size()) {
        sum += x[i] * x[i];
    }

    sum = cg::reduce(warp, sum, cg::plus<dtype>()); // sum of all elements in the row

    // dtype mean = sum / hidden_size;

    // sum = 0.0f;
    dtype rms = sqrtf(sum / hidden_size + epsilon);

    dtype *o = output + idx * hidden_size;
    for (int i = warp.thread_rank(); i < hidden_size; i += warp.size()) {
        o[i] = x[i] * weight[i] / rms;
    }
}

template <typename dtype>
void rms_norm_v1(dtype *output, dtype *input, dtype *weight, float epsilon, int hidden_size, int num_tokens) {
    int blockSize = 256;
    int gridSize = div_ceil(num_tokens * 32, blockSize);
    rms_norm_kernel_v1<<<gridSize, blockSize>>>(output, input, weight, epsilon, hidden_size, num_tokens);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/********************************************************* rms_norm ****************************************************/

template<typename dtype>
void CUDA<dtype>::rms_norm(dtype *output, dtype *input, dtype *weight, float epsilon, int hidden_size, int num_tokens) {
    rms_norm_v0(output, input, weight, epsilon, hidden_size, num_tokens);
    // rms_norm_v1(output, input, weight, epsilon, hidden_size, num_tokens);
}
