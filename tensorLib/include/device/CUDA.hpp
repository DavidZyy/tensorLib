#pragma once

#include "Device.hpp"
#include <cstddef>
#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cublas_v2.h>

template <typename dtype>
class CUDA : public Device<dtype> {
public:
    CUDA() : Device<dtype>(0), data_(nullptr) {}
    CUDA(size_t size, dtype* ptr) : Device<dtype>(size), data_(ptr) {}
    CUDA(size_t size);
    ~CUDA() override;

    void matmul(const dtype* lhs, const dtype* rhs, dtype* result, 
        const std::vector<int>& lhs_stride, 
        const std::vector<int>& rhs_stride, 
        size_t lhs_offset,
        size_t rhs_offset,
        const std::vector<int>& result_shape,
        size_t result_elements,
        size_t K) override;
 
    dtype* getDataPtr() override { return data_; }
    void full (size_t num_elements, dtype fill_value) override;
    dtype getDataLinear(size_t liner_index) const override;
    void contiguous(
        dtype* result, 
        const std::vector<int>& shape,
        const std::vector<int>& stride, 
        size_t offset,
        size_t num_elements) override;

    void setItemEwise(
        dtype* src,
        const std::vector<int>& shape,
        const std::vector<int>& stride,
        size_t offset,
        size_t num_elements) override;

    void setItemScalar(
        dtype value,
        const std::vector<int>& shape,
        const std::vector<int>& stride,
        size_t offset,
        size_t num_elements) override;
// private:
    dtype *data_;
};

__device__ inline size_t convertIdx(
    size_t linear_index, 
    const int* shape, 
    const int* stride, 
    size_t offset, 
    int dim_size) 
{
    size_t linear_index_new = 0;
    
    for (int i = dim_size - 1; i >= 0; --i) {
        int cur_dim_id = linear_index % shape[i];
        linear_index /= shape[i];
        linear_index_new += cur_dim_id * stride[i];
    }
    
    return linear_index_new + offset;
}

#define CUDA_CHECK(call)                                                    \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", ";      \
        std::cerr << "code: " << error << ", reason: " << cudaGetErrorString(error) << std::endl; \
        exit(1);                                                            \
    }                                                                       \
}

#define CUBLAS_CHECK(call)                                                  \
{                                                                           \
    const cublasStatus_t status = call;                                     \
    if (status != CUBLAS_STATUS_SUCCESS)                                    \
    {                                                                       \
        std::cerr << "CUBLAS Error: " << __FILE__ << ":" << __LINE__ << ", "; \
        std::cerr << "status: " << cublasGetErrorString(status) << std::endl; \
        exit(1);                                                            \
    }                                                                       \
}

inline std::string cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                             return "Unknown cuBLAS error";
    }
}
