#include "CUDA.hpp"
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <library_types.h>
#include <iostream>
#include <vector>

template class CUDA<float>;
template class CUDA<int>;

template <typename dtype>
CUDA<dtype>::CUDA(size_t size) : Device<dtype>(size) {
    CUDA_CHECK(cudaMalloc(&this->data_, size * sizeof(dtype)));
}

template <typename dtype>
CUDA<dtype>::~CUDA() {
    CUDA_CHECK(cudaFree(this->data_));
}

/**
 * I think this implementation is not efficient, but it should be good enough for now.
 * We can improve it later.
 * Maybe we can make lhs and rhs be contiguous, and use shared memory in one block ?
 * execuate contiguous in cpu is not efficient, so i give up this step when perform batched matmul in cpu,
 * but in cuda, maybe we can do it ?
 *
 * @tparam dtype 
 */
template <typename dtype>
__global__ void matmulKernel(const dtype* lhs, const dtype* rhs, dtype* result, 
                             CudaVec lhs_stride, CudaVec rhs_stride, 
                             size_t lhs_offset, size_t rhs_offset,
                             CudaVec result_shape, size_t result_elements,
                             size_t K) 
{
    size_t ndim = result_shape.size;
    // Global thread index for each result element
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= result_elements) return;

    size_t linear_index = idx;
    size_t Aoff = lhs_offset, Boff = rhs_offset;

    // Compute offsets for lhs and rhs
    for (int i = ndim - 1; i >= 0; --i) {
        int cur_dim_id = linear_index % result_shape.data[i];
        linear_index /= result_shape.data[i];

        if (i != ndim - 1)
            Aoff += cur_dim_id * lhs_stride.data[i];
        if (i != ndim - 2)
            Boff += cur_dim_id * rhs_stride.data[i];
    }

    // Compute the dot product
    dtype sum = 0;
    int t1 = lhs_stride.data[ndim - 1], t2 = rhs_stride.data[ndim - 2];
    for (int k = 0; k < K; ++k) {
        sum += lhs[Aoff + k * t1] * rhs[Boff + k * t2];
    }

    // Store the result
    result[idx] = sum;
}

// Wrapper function to launch the CUDA kernel
template <typename dtype>
void CUDA<dtype>::matmul(const dtype* lhs, const dtype* rhs, dtype* result, 
    const std::vector<int>& lhs_stride, 
    const std::vector<int>& rhs_stride, 
    size_t lhs_offset,
    size_t rhs_offset,
    const std::vector<int>& result_shape,
    size_t result_elements,
    size_t K)
{
    // Launch the kernel
    int threads_per_block = 256;
    int blocks = (result_elements + threads_per_block - 1) / threads_per_block;
    matmulKernel<<<blocks, threads_per_block>>>(lhs, rhs, result, 
                                                VecToCuda(lhs_stride), VecToCuda(rhs_stride), 
                                                lhs_offset, rhs_offset, 
                                                VecToCuda(result_shape), result_elements, K);
    CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename dtype>
dtype CUDA<dtype>::getDataLinear(size_t linear_index) const {
    dtype result;
    CUDA_CHECK(cudaMemcpy(&result, this->data_ + linear_index, sizeof(dtype), cudaMemcpyDeviceToHost));
    return result;
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

// General template for CUDA unary operations
template <typename dtype, dtype (*op)(dtype)>
void applyUnaryOperation(dtype* result, dtype* src, size_t num_elements) {
    int blockSize = 256;  // Number of threads per block (adjust based on optimization needs)
    int gridSize = (num_elements + blockSize - 1) / blockSize;  // Number of blocks
    unaryKernel<dtype, op><<<gridSize, blockSize>>>(result, src, num_elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename dtype>
__device__ dtype negateFunc(dtype x) {
    return -x;
}

template <typename dtype>
void CUDA<dtype>::neg(dtype* result, size_t num_elements) {
    applyUnaryOperation<dtype, negateFunc<dtype>>(result, this->data_, num_elements);
}

template <typename dtype>
__device__ dtype sinFunc(dtype x) {
    if constexpr (std::is_same<dtype, int>::value) {
        // If dtype is an integer type, cast x to float and calculate sine, or it will link to std::sin, which is not supported on CUDA
        return static_cast<dtype>(sin(static_cast<float>(x)));
    } else {
        return sin(x);
    }
}

template <typename dtype>
void CUDA<dtype>::sin(dtype* result, size_t num_elements) {
    applyUnaryOperation<dtype, sinFunc<dtype>>(result, this->data_, num_elements);
}

template <typename dtype>
__device__ dtype cosFunc(dtype x) {
    if constexpr (std::is_same<dtype, int>::value) {
        return static_cast<dtype>(cos(static_cast<float>(x)));
    } else {
        return cos(x);
    }
}

template <typename dtype>
void CUDA<dtype>::cos(dtype* result, size_t num_elements) {
    applyUnaryOperation<dtype, cosFunc<dtype>>(result, this->data_, num_elements);
}

template <typename dtype>
__device__ dtype expFunc(dtype x) {
    if constexpr (std::is_same<dtype, int>::value) {
        return static_cast<dtype>(exp(static_cast<float>(x)));
    } else {
        return exp(x);
    }
}

template <typename dtype>
void CUDA<dtype>::exp(dtype* result, size_t num_elements) {
    applyUnaryOperation<dtype, expFunc<dtype>>(result, this->data_, num_elements);
}

template <typename dtype>
__device__ dtype logFunc(dtype x) {
    if constexpr (std::is_same<dtype, int>::value) {
        return static_cast<dtype>(log(static_cast<float>(x)));
    } else {
        return log(x);
    }
}

template <typename dtype>
void CUDA<dtype>::log(dtype* result, size_t num_elements) {
    applyUnaryOperation<dtype, logFunc<dtype>>(result, this->data_, num_elements);
}

template <typename dtype>
__device__ dtype absFunc(dtype x) {
    return abs(x);
}

template <typename dtype>
void CUDA<dtype>::abs(dtype* result, size_t num_elements) {
    applyUnaryOperation<dtype, absFunc<dtype>>(result, this->data_, num_elements);
}

template <typename dtype>
__device__ dtype tanhFunc(dtype x) {
    if constexpr (std::is_same<dtype, int>::value) {
        return static_cast<dtype>(tanh(static_cast<float>(x)));
    } else {
        return tanh(x);
    }
}

template <typename dtype>
void CUDA<dtype>::tanh(dtype* result, size_t num_elements) {
    applyUnaryOperation<dtype, tanhFunc<dtype>>(result, this->data_, num_elements);
}

template <typename dtype>
__device__ dtype siluFunc(dtype x) {
    if constexpr (std::is_same<dtype, int>::value) {
        return static_cast<dtype>(static_cast<float>(x) * (1 / (1 + exp(-static_cast<float>(x)))));
    } else {
        return x * (1 / (1 + exp(-x)));
    }
}

template <typename dtype>
void CUDA<dtype>::silu(dtype* result, size_t num_elements) {
    applyUnaryOperation<dtype, siluFunc<dtype>>(result, this->data_, num_elements);
}

template <typename dtype>
__device__ dtype sqrtFunc(dtype x) {
    if (x >= 0) {
        if constexpr (std::is_same<dtype, int>::value) {
            return static_cast<dtype>(sqrt(static_cast<float>(x)));
        } else {
            return sqrt(x); // Rsqrt calculation
        }
    } else {
        return nan("");
    }
}

template <typename dtype>
void CUDA<dtype>::sqrt(dtype* result, size_t num_elements) {
    applyUnaryOperation<dtype, sqrtFunc<dtype>>(result, this->data_, num_elements);
}

template <typename dtype>
__device__ dtype rsqrtFunc(dtype x) {
    if (x > 0) {
        if constexpr (std::is_same<dtype, int>::value) {
            return static_cast<dtype>(rsqrt(static_cast<float>(x)));
        } else {
            return rsqrt(x); // Rsqrt calculation
        }
    } else {
        return nan("");
    }
}

template <typename dtype>
void CUDA<dtype>::rsqrt(dtype* result, size_t num_elements) {
    applyUnaryOperation<dtype, rsqrtFunc<dtype>>(result, this->data_, num_elements);
}

////////////////////////////////////////////////////// binary operations ///////////////////////////////////////////////////////////////////////////////
template <typename dtype> static inline __device__ dtype addFunc(dtype x, dtype y) { return x + y; }
template <typename dtype> static inline __device__ dtype subFunc(dtype x, dtype y) { return x - y; }
template <typename dtype> static inline __device__ dtype mulFunc(dtype x, dtype y) { return x * y; }
template <typename dtype> static inline __device__ dtype divFunc(dtype a, dtype b) {
    if (b == 0) {
        return nan("");
    }
    return a / b;
}
template <typename dtype> static inline __device__ dtype powFunc(dtype a, dtype b) { return pow(a, b); }

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
    CUDA_CHECK(cudaDeviceSynchronize());
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
    CUDA_CHECK(cudaDeviceSynchronize());
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

////////////////////////////////////////////////////// reduce operations ///////////////////////////////////////////////////////////////////////////////
/**
 * maybe could use parallel reduction algorithm to parallelize reduce_size, this will faster. ( O(n)-> O(log(n)) )
 *
 * @brief Reduce operation kernel
 */
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
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename dtype, bool (*comp)(dtype, dtype)>
__global__ void reduceArgKernel(int* result, const dtype* data, size_t reduce_size, size_t num_elements) {
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

template <typename dtype>
template <bool (*comp)(dtype, dtype)>
void CUDA<dtype>::reduceOperationArg(int* result, size_t reduce_size, size_t num_elements) const {
    int blockSize = 256;  // Number of threads per block (adjust based on optimization needs)
    int gridSize = (num_elements / reduce_size + blockSize - 1) / blockSize;  // Number of blocks

    reduceArgKernel<dtype, comp><<<gridSize, blockSize>>>(result, this->data_, reduce_size, num_elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename dtype> static inline __device__ dtype maxFunc(dtype a, dtype b) { return max(a, b); }
template <typename dtype> static inline __device__ dtype minFunc(dtype a, dtype b) { return min(a, b); }
template <typename dtype> static inline __device__ dtype sumFunc(dtype a, dtype b) { return a + b; }
template <typename dtype> static inline __device__ bool argmaxFunc(dtype a, dtype b) { return a > b; }
template <typename dtype> static inline __device__ bool argminFunc(dtype a, dtype b) { return a < b; }

template <typename dtype> void CUDA<dtype>::max(dtype* result, size_t reduce_size, size_t num_elements) const { reduceOperation<maxFunc<dtype>>(result, reduce_size, num_elements); }
template <typename dtype> void CUDA<dtype>::min(dtype* result, size_t reduce_size, size_t num_elements) const { reduceOperation<minFunc<dtype>>(result, reduce_size, num_elements); }
template <typename dtype> void CUDA<dtype>::sum(dtype* result, size_t reduce_size, size_t num_elements) const { reduceOperation<sumFunc<dtype>>(result, reduce_size, num_elements); }
template <typename dtype> void CUDA<dtype>::argmax(int* result, size_t reduce_size, size_t num_elements) const { reduceOperationArg<argmaxFunc<dtype>>(result, reduce_size, num_elements); }
template <typename dtype> void CUDA<dtype>::argmin(int* result, size_t reduce_size, size_t num_elements) const { reduceOperationArg<argminFunc<dtype>>(result, reduce_size, num_elements); }
