#include "device/CUDA.hpp"
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <library_types.h>
#include <iostream>
#include <vector>
#include <curand_kernel.h>

template class CUDA<int8_t>;
template class CUDA<half>;
template class CUDA<float>;
template class CUDA<int>;

template <typename dtype>
CUDA<dtype>::CUDA(size_t size) : Device<dtype>(size) {
    CUDA_CHECK(cudaMalloc(&this->data_, size * sizeof(dtype)));
}

// template <typename dtype>
// CUDA<dtype>::CUDA(size_t size) : Device<dtype>(size) {
//     // Allocate memory aligned to 16 bytes
//     CUDA_CHECK(cudaMallocAligned(&this->data_, size * sizeof(dtype), 16));
// }

template <typename dtype>
CUDA<dtype>::~CUDA() {
    // free a null prt get error: code: 4, reason: driver shutting down
    if(this->data_ != nullptr)
        CUDA_CHECK(cudaFree(this->data_));
}

template <typename dtype>
__global__ void fullKernel(dtype* data, size_t num_elements, dtype fill_value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = static_cast<dtype>(fill_value);
    }
}

template <typename dtype>
void CUDA<dtype>::full(size_t num_elements, dtype fill_value) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    fullKernel<<<blocks_per_grid, threads_per_block>>>(this->data_, num_elements, fill_value);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename dtype>
__global__ void randnKernel(dtype* data, size_t num_elements, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        curandState state;
        unsigned long long thread_seed = seed + idx + clock64(); // Timestamp + idx to ensure unique seeds
        curand_init(thread_seed + idx, idx, 0, &state);
        data[idx] = curand_normal(&state);
    }
}

template <typename dtype>
void CUDA<dtype>::randn(size_t num_elements) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    unsigned long long seed = time(NULL); // Get a time-based seed
    randnKernel<<<blocks_per_grid, threads_per_block>>>(this->data_, num_elements, seed);
}

template <typename dtype>
dtype CUDA<dtype>::getDataLinear(size_t linear_index) const {
    dtype result;
    CUDA_CHECK(cudaMemcpy(&result, this->data_ + linear_index, sizeof(dtype), cudaMemcpyDeviceToHost));
    return result;
}

template <typename dtype>
void CUDA<dtype>::setDataLinear(size_t linear_index, dtype value) {
    CUDA_CHECK(cudaMemcpy(this->data_ + linear_index, &value, sizeof(dtype), cudaMemcpyHostToDevice));
}

template <typename dtype>
__global__ void contiguous_kernel(
    dtype* result,
    const dtype* data,
    CudaVec shape,
    CudaVec stride,
    size_t offset,
    size_t num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        size_t linear_index_new = convertIdx(i, shape, stride, offset);
 
        result[i] = data[linear_index_new];
    }
}

template <typename dtype>
void CUDA<dtype>::contiguous(
    dtype* result,
    const std::vector<int>& shape,
    const std::vector<int>& stride,
    size_t offset,
    size_t num_elements) 
{
    // Calculate grid and block dimensions
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    contiguous_kernel<<<num_blocks, threads_per_block>>>(
        result, this->data_, VecToCuda(shape), VecToCuda(stride), offset, num_elements);
    CUDA_CHECK(cudaGetLastError());
}

template <typename dtype>
__global__ void setItemEwiseKernel(
    dtype* data,
    const dtype* src,
    CudaVec shape,
    CudaVec stride, 
    size_t offset,
    size_t num_elements) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        size_t linearIdx = convertIdx(i, shape, stride, offset);
        data[linearIdx] = src[i];
    }
}

template <typename dtype>
void CUDA<dtype>::setItemEwise(
    dtype* src,
    const std::vector<int>& shape,
    const std::vector<int>& stride,
    size_t offset,
    size_t num_elements) 
{
    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (num_elements + blockSize - 1) / blockSize;

    // Launch the kernel
    setItemEwiseKernel<<<gridSize, blockSize>>>(
        this->data_, src, VecToCuda(shape), VecToCuda(stride), offset, num_elements);

    CUDA_CHECK(cudaGetLastError());
}

template <typename dtype>
__global__ void setItemScalarKernel(
    dtype* data,
    const dtype value,
    CudaVec shape,
    CudaVec stride, 
    size_t offset,
    size_t num_elements) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        size_t linearIdx = convertIdx(i, shape, stride, offset);
        data[linearIdx] = value;
    }
}

template <typename dtype>
void CUDA<dtype>::setItemScalar(
    dtype value,
    const std::vector<int>& shape,
    const std::vector<int>& stride,
    size_t offset,
    size_t num_elements) 
{
    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (num_elements + blockSize - 1) / blockSize;

    // Launch the kernel
    setItemScalarKernel<<<gridSize, blockSize>>>(
        this->data_, value, VecToCuda(shape), VecToCuda(stride), offset, num_elements);

    CUDA_CHECK(cudaGetLastError());
}

////////////////////////////////////////////////////// unary operations ///////////////////////////////////////////////////////////////////////////////
/**
 * pass function pointer like below have bug,
 * __global__ void unaryKernel(dtype* result, const dtype* src, size_t num_elements, dtype (*op)(dtype)),
 * seems should use cudaMemcpyFromSymbol first, so I use template instead.
 */
template <typename dtype, dtype (*op)(dtype)>
__global__ void unaryKernel(dtype* result, const dtype* src, size_t num_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        result[i] = op(src[i]);
    }
}

template <typename dtype>
template <dtype (*op)(dtype)>
void CUDA<dtype>::applyUnaryOperation(dtype* result, size_t num_elements) const {
    int blockSize = 256;  // Number of threads per block (adjust based on optimization needs)
    int gridSize = (num_elements + blockSize - 1) / blockSize;  // Number of blocks
    unaryKernel<dtype, op><<<gridSize, blockSize>>>(result, this->data_, num_elements);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename dtype>
__device__ dtype negateFunc(dtype x) {
    return -x;
}

template <typename dtype>
void CUDA<dtype>::neg(dtype* result, size_t num_elements) {
    applyUnaryOperation<negateFunc<dtype>>(result, num_elements);
}

// template <typename dtype>
// __device__ dtype sinFunc(dtype x) {
//     if constexpr (std::is_same<dtype, int>::value || std::is_same<dtype, int8_t>::value) {
//         // If dtype is an integer type, cast x to float and calculate sine, or it will link to std::sin, which is not supported on CUDA
//         return static_cast<dtype>(sin(static_cast<float>(x)));
//     } else {
//         return sin(x);
//     }
// }

template <typename dtype>
__device__ dtype sinFunc(dtype x) {
    if constexpr (std::is_same<dtype, int>::value || std::is_same<dtype, int8_t>::value) {
        // If dtype is an integer type, cast x to float and calculate sine, or it will link to std::sin, which is not supported on CUDA
        return static_cast<dtype>(sin(static_cast<float>(x)));
    } else if constexpr (std::is_same<dtype, __half>::value) {
        // If dtype is __half (half precision), use the CUDA-specific __hsin
        return __float2half(sinf(__half2float(x)));
    } else {
        // For other types (float, double), use standard sin
        return sin(x);
    }
}


template <typename dtype>
void CUDA<dtype>::sin(dtype* result, size_t num_elements) {
    applyUnaryOperation<sinFunc<dtype>>(result, num_elements);
}

template <typename dtype>
__device__ dtype cosFunc(dtype x) {
    if constexpr (std::is_same<dtype, int>::value || std::is_same<dtype, int8_t>::value) {
        return static_cast<dtype>(cos(static_cast<float>(x)));
    } else if constexpr (std::is_same<dtype, __half>::value) {
        // If dtype is __half (half precision), use the CUDA-specific __hsin
        return __float2half(cosf(__half2float(x)));
    } else {
        return cos(x);
    }
}

template <typename dtype>
void CUDA<dtype>::cos(dtype* result, size_t num_elements) {
    applyUnaryOperation<cosFunc<dtype>>(result, num_elements);
}

template <typename dtype>
__device__ dtype expFunc(dtype x) {
    if constexpr (std::is_same<dtype, int>::value || std::is_same<dtype, int8_t>::value) {
        return static_cast<dtype>(exp(static_cast<float>(x)));
    } else if constexpr (std::is_same<dtype, __half>::value) {
        // If dtype is __half (half precision), use the CUDA-specific __hsin
        return __float2half(expf(__half2float(x)));
    } else {
        return exp(x);
    }
}

template <typename dtype>
void CUDA<dtype>::exp(dtype* result, size_t num_elements) {
    applyUnaryOperation<expFunc<dtype>>(result, num_elements);
}

template <typename dtype>
__device__ dtype logFunc(dtype x) {
    if constexpr (std::is_same<dtype, int>::value || std::is_same<dtype, int8_t>::value) {
        return static_cast<dtype>(log(static_cast<float>(x)));
    } else if constexpr (std::is_same<dtype, __half>::value) {
        // If dtype is __half (half precision), use the CUDA-specific __hsin
        return __float2half(logf(__half2float(x)));
    } else {
        return log(x);
    }
}

template <typename dtype>
void CUDA<dtype>::log(dtype* result, size_t num_elements) {
    applyUnaryOperation<logFunc<dtype>>(result, num_elements);
}

template <typename dtype>
__device__ dtype absFunc(dtype x) {
    if constexpr (std::is_same<dtype, __half>::value) {
        return __float2half(abs(__half2float(x)));
    } else {
        return abs(x);
    }
}

template <typename dtype>
void CUDA<dtype>::abs(dtype* result, size_t num_elements) {
    applyUnaryOperation<absFunc<dtype>>(result, num_elements);
}

template <typename dtype>
__device__ dtype tanhFunc(dtype x) {
    if constexpr (std::is_same<dtype, int>::value || std::is_same<dtype, int8_t>::value) {
        return static_cast<dtype>(tanh(static_cast<float>(x)));
    } else if constexpr (std::is_same<dtype, __half>::value) {
        // If dtype is __half (half precision), use the CUDA-specific __hsin
        return __float2half(tanhf(__half2float(x)));
    }  else {
        return tanh(x);
    }
}

template <typename dtype>
void CUDA<dtype>::tanh(dtype* result, size_t num_elements) {
    applyUnaryOperation<tanhFunc<dtype>>(result, num_elements);
}

template <typename dtype>
__device__ dtype siluFunc(dtype x) {
    // if constexpr (std::is_same<dtype, int>::value || std::is_same<dtype, int8_t>::value) {
    if constexpr (std::is_same<dtype, int>::value || std::is_same<dtype, int8_t>::value || std::is_same_v<dtype, half>) {
        return static_cast<dtype>(static_cast<float>(x) * (1 / (1 + exp(-static_cast<float>(x)))));
    } else {
        return x * (1 / (1 + exp(-x)));
    }
}

template <typename dtype>
void CUDA<dtype>::silu(dtype* result, size_t num_elements) {
    applyUnaryOperation<siluFunc<dtype>>(result, num_elements);
}

template <typename dtype>
__device__ dtype sqrtFunc(dtype x) {
    if constexpr (std::is_same<dtype, __half>::value) {
        // For __half, first convert to float for comparison
        if (__half2float(x) >= 0) {
            return __float2half(sqrtf(__half2float(x)));
        } else {
            return __float2half(nanf("")); // Return NaN for negative input
        }
    } else {
        if (x >= 0) {
            if constexpr (std::is_same<dtype, int>::value || std::is_same<dtype, int8_t>::value) {
                return static_cast<dtype>(sqrt(static_cast<float>(x)));
            } else {
                return sqrt(x); // Rsqrt calculation
            }
        } else {
            return nan("");
        }
    }
}

template <typename dtype>
void CUDA<dtype>::sqrt(dtype* result, size_t num_elements) {
    applyUnaryOperation<sqrtFunc<dtype>>(result, num_elements);
}

template <typename dtype>
__device__ dtype rsqrtFunc(dtype x) {
    if constexpr (std::is_same<dtype, __half>::value) {
        // For __half, first convert to float for comparison
        if (__half2float(x) > 0) {
            return __float2half(rsqrt(__half2float(x)));
        } else {
            return __float2half(nanf("")); // Return NaN for negative input
        }
    } else { 
        if (x > 0) {
            if constexpr (std::is_same<dtype, int>::value || std::is_same<dtype, int8_t>::value) {
                return static_cast<dtype>(rsqrt(static_cast<float>(x)));
            } else {
                return rsqrt(x); // Rsqrt calculation
            }
        } else {
            return nan("");
        }
    }
}

template <typename dtype>
void CUDA<dtype>::rsqrt(dtype* result, size_t num_elements) {
    applyUnaryOperation<rsqrtFunc<dtype>>(result, num_elements);
}

////////////////////////////////////////////////////// binary operations ///////////////////////////////////////////////////////////////////////////////
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

template <typename dtype, dtype (*op)(dtype, dtype)>
__global__ void binaryKernel(dtype* result, const dtype* src1, const dtype* src2, size_t num_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        result[i] = op(src1[i], src2[i]);
    }
}

template <typename dtype>
template <dtype (*op)(dtype, dtype)>
void CUDA<dtype>::applyBinaryOperation(dtype* result,  const dtype* other, size_t num_elements) const {
    int blockSize = 256;  // Number of threads per block (adjust based on optimization needs)
    int gridSize = (num_elements + blockSize - 1) / blockSize;  // Number of blocks
    binaryKernel<dtype, op><<<gridSize, blockSize>>>(result, this->data_, other, num_elements);
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

template <typename dtype>
template <dtype (*op)(dtype, dtype)>
void CUDA<dtype>::applyBinaryScalarOperation(dtype* result,  dtype value, size_t num_elements) const {
    int blockSize = 256;  // Number of threads per block (adjust based on optimization needs)
    int gridSize = (num_elements + blockSize - 1) / blockSize;  // Number of blocks
    binaryScalarKernel<dtype, op><<<gridSize, blockSize>>>(result, this->data_, value, num_elements);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename dtype> void CUDA<dtype>::add(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<addFunc<dtype>>(result, other, num_elements);}
template <typename dtype> void CUDA<dtype>::sub(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<subFunc<dtype>>(result, other, num_elements);}
template <typename dtype> void CUDA<dtype>::mul(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<mulFunc<dtype>>(result, other, num_elements);}
template <typename dtype> void CUDA<dtype>::div(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<divFunc<dtype>>(result, other, num_elements);}

template <typename dtype> void CUDA<dtype>::add(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<addFunc<dtype>>(result, value, num_elements);}
template <typename dtype> void CUDA<dtype>::sub(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<subFunc<dtype>>(result, value, num_elements);}
template <typename dtype> void CUDA<dtype>::mul(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<mulFunc<dtype>>(result, value, num_elements);}
template <typename dtype> void CUDA<dtype>::div(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<divFunc<dtype>>(result, value, num_elements);}
template <typename dtype> void CUDA<dtype>::pow(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<powFunc<dtype>>(result, value, num_elements);}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// template <typename dtype>
// __global__ void apply_rotary_emb_kernel(const dtype* input, dtype* result, int start_pos, int H, int W) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;  // Row index
//     int j = (blockIdx.y * blockDim.y + threadIdx.y) * 2;  // Column index (step by 2 for paired elements)
// 
//     if (i < H && j < W) {
//         int offset = i * W;
//         dtype theta = start_pos * 1.0f / pow(10000.0f, static_cast<dtype>(j) / static_cast<dtype>(W));
//         dtype cos_theta = cosf(theta); // only accept float for now
//         dtype sin_theta = sinf(theta);
// 
//         dtype v0 = input[offset + j];
//         dtype v1 = input[offset + j + 1];
// 
//         dtype rotary_emb_real = v0 * cos_theta - v1 * sin_theta;
//         dtype rotary_emb_imag = v0 * sin_theta + v1 * cos_theta;
// 
//         result[offset + j] = rotary_emb_real;
//         result[offset + j + 1] = rotary_emb_imag;
//     }
// }
// 
// template <typename dtype>
// void CUDA<dtype>::apply_rotary_emb(const dtype* input, dtype* result, int start_pos, int H, int W) const {
//     dim3 threadsPerBlock(16, 16);  // Define block size (16x16 is a typical choice, can be adjusted)
//     dim3 numBlocks((H + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                    (W / 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);  // Divide by 2 for W because j increments by 2
// 
//     apply_rotary_emb_kernel<<<numBlocks, threadsPerBlock>>>(input, result, start_pos, H, W);
//     CUDA_CHECK(cudaGetLastError());
//     // CUDA_CHECK(cudaDeviceSynchronize());
// }

template <typename dtype>
__global__ void apply_rotary_emb_kernel(const dtype* input, dtype* result, int start_pos, int B, int T, int n_heads, int head_dim) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int h = blockIdx.z;
    int d = threadIdx.x * 2;

    int offset = b * T * n_heads * head_dim + t * n_heads * head_dim + h * head_dim + d;

    float theta = (start_pos + t) * 1.0f / pow(10000.0f, static_cast<float>(d) / static_cast<float>(head_dim));
    // dtype theta = (start_pos + t) * 1.0f / pow(10000.0f, static_cast<dtype>(d) / static_cast<dtype>(head_dim));
    // dtype theta;
    // if constexpr (std::is_same_v<dtype, half>) {
    //     theta = (start_pos + t) * 1.0f / pow(10000.0f, __half2float(static_cast<float>(d)) / __half2float(static_cast<float>(head_dim)));
    // } else {
    //     theta = (start_pos + t) * 1.0f / pow(10000.0f, static_cast<dtype>(d) / static_cast<dtype>(head_dim));
    // }

    // dtype cos_theta = cosf(theta); // only accept float for now
    // dtype sin_theta = sinf(theta);

    dtype cos_theta = static_cast<dtype>(cos(theta));
    dtype sin_theta = static_cast<dtype>(sin(theta));

    dtype v0 = input[offset];
    dtype v1 = input[offset + 1];

    dtype rotary_emb_real = v0 * cos_theta - v1 * sin_theta;
    dtype rotary_emb_imag = v0 * sin_theta + v1 * cos_theta;

    result[offset] = rotary_emb_real;
    result[offset + 1] = rotary_emb_imag;
}

template <typename dtype>
void CUDA<dtype>::apply_rotary_emb(const dtype* input, dtype* result, int start_pos, int B, int T, int n_heads, int head_dim) const {
    dim3 grid(B, T, n_heads);
    dim3 block(head_dim / 2);

    apply_rotary_emb_kernel<<<grid, block>>>(input, result, start_pos, B, T, n_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

// CUDA kernel for type casting
template <typename dtype, typename OtherType>
__global__ void type_cast_kernel(dtype* result, const OtherType* src, size_t num_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        if constexpr (std::is_same_v<dtype, half>) {
            result[i] = __float2half(static_cast<float>(src[i]));
        } else {
            result[i] = static_cast<dtype>(src[i]);
        }
    }
}

template <typename dtype>
template <typename OtherType>
void CUDA<dtype>::type_cast(dtype* result, const OtherType* src, size_t num_elements) {
    // Define block and grid sizes
    size_t blockSize = 256;
    size_t numBlocks = (num_elements + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    type_cast_kernel<<<numBlocks, blockSize>>>(result, src, num_elements);

    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}
// Explicit instantiation of the template function for specific types
template void CUDA<float>::type_cast<float>(float*, const float*, size_t);
template void CUDA<float>::type_cast<int>(float*, const int*, size_t);
template void CUDA<float>::type_cast<half>(float*, const half*, size_t);

template void CUDA<int>::type_cast<float>(int*, const float*, size_t);
template void CUDA<int>::type_cast<int>(int*, const int*, size_t);
template void CUDA<int>::type_cast<half>(int*, const half*, size_t);

template void CUDA<half>::type_cast<float>(half*, const float*, size_t);
template void CUDA<half>::type_cast<int>(half*, const int*, size_t);
template void CUDA<half>::type_cast<half>(half*, const half*, size_t);

