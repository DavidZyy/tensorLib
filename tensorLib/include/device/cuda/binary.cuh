#include "device/cuda/CUDA.cuh"

template <typename dtype> static inline __device__ dtype addFunc(dtype x, dtype y) { 
    if constexpr (std::is_same_v<dtype, __half>) {
        return __hadd(x, y); // Use CUDA's __haddfor addition}
    }
    return x + y; 
}
template <typename dtype> static inline __device__ dtype subFunc(dtype x, dtype y) { return x - y; }
template <typename dtype> static inline __device__ dtype mulFunc(dtype x, dtype y) { 
    if constexpr (std::is_same<dtype, __half>::value) {
        return __hmul(x, y); // Use CUDA's __hmul for multiplication
    }
    return x * y; 
}
template <typename dtype> static inline __device__ dtype divFunc(dtype a, dtype b) {
    // Handle comparison for __half type
    if constexpr (std::is_same<dtype, __half>::value) {
        if (__heq(b, __float2half(0.0f))) { // Use CUDA's __heq for comparison
            return __float2half(nanf("")); // Return NaN for division by zero
        }
        return __hdiv(a, b); // Use CUDA's __hdiv for division
    }
    // Handle comparison for other types
    else if (b == 0) {
        return nan(""); // Return NaN for division by zero
    }
    return a / b; // Default division for other types
}

// template <typename dtype> static inline __device__ dtype powFunc(dtype a, dtype b) { return pow(a, b); }
template <typename dtype> static inline __device__ dtype powFunc(dtype a, dtype b) { 
    if constexpr (std::is_same<dtype, __half>::value) {
        return __float2half(pow(__half2float(a), __half2float(b))); 
    } else {
        return pow(a, b); 
    }
}

/********************************************************** kernels **************************************************************** */

template <typename dtype, dtype (*op)(dtype, dtype)>
__global__ void binaryKernel(dtype* result, const dtype* src1, const dtype* src2, size_t num_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        result[i] = op(src1[i], src2[i]);
    }
}

template <typename dtype, dtype (*op)(dtype, dtype)>
void applyBinaryOperation(dtype* result,  const dtype* lhs, const dtype* rhs, size_t num_elements) {
    int blockSize = 256;  // Number of threads per block (adjust based on optimization needs)
    int gridSize = (num_elements + blockSize - 1) / blockSize;  // Number of blocks
    binaryKernel<dtype, op><<<gridSize, blockSize>>>(result, lhs, rhs, num_elements);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename dtype, dtype (*op)(dtype, dtype)>
__global__ void binaryScalarKernel(dtype* result, const dtype* src1, dtype value, size_t num_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        result[i] = op(src1[i], value);
    }
}

template <typename dtype, dtype (*op)(dtype, dtype)>
void applyBinaryScalarOperation(dtype* result,  dtype* input, dtype value, size_t num_elements) {
    int blockSize = 256;  // Number of threads per block (adjust based on optimization needs)
    int gridSize = (num_elements + blockSize - 1) / blockSize;  // Number of blocks
    binaryScalarKernel<dtype, op><<<gridSize, blockSize>>>(result, input, value, num_elements);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}
