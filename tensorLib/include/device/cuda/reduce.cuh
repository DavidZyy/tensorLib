#include "device/cuda/CUDA.cuh"

template <typename dtype> 
static inline __device__ dtype maxFunc(dtype a, dtype b) {
    return max(static_cast<float>(a), static_cast<float>(b)); 
}
template <typename dtype> 
static inline __device__ dtype minFunc(dtype a, dtype b) {
    return min(static_cast<float>(a), static_cast<float>(b)); 
}
template <typename dtype> 
static inline __device__ dtype sumFunc(dtype a, dtype b) {
    return a + b; 
}

/********************************************************** kernels **************************************************************** */

template <typename dtype, dtype (*op)(dtype, dtype)>
__global__ void reduceKernel(dtype* result, const dtype* data, size_t reduce_size, size_t num_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements / reduce_size) {
        dtype temp = data[i * reduce_size];
        for (int j = 1; j < reduce_size; j++) {
            temp = op(temp, data[i * reduce_size + j]);
        }
        result[i] = temp;
    }
}

template <typename dtype>
template <dtype (*op)(dtype, dtype)>
void CUDA<dtype>::reduceOperation(dtype* result, size_t reduce_size, size_t num_elements) const {
    int blockSize = 256;  // Number of threads per block (adjust based on optimization needs)
    int gridSize = (num_elements / reduce_size + blockSize - 1) / blockSize;  // Number of blocks

    reduceKernel<dtype, op><<<gridSize, blockSize>>>(result, this->data_, reduce_size, num_elements);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}
