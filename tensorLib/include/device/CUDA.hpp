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

    // batched matmul
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

    // unary operations
    void neg(dtype* result, size_t num_elements) override;
    void sin(dtype* result, size_t num_elements) override;
    void cos(dtype* result, size_t num_elements)   override;
    void exp(dtype* result, size_t num_elements)   override;
    void log(dtype* result, size_t num_elements)   override;
    void abs(dtype* result, size_t num_elements)   override;
    void tanh(dtype* result, size_t num_elements)  override;
    void silu(dtype* result, size_t num_elements)  override;
    void sqrt(dtype* result, size_t num_elements)  override;
    void rsqrt(dtype* result, size_t num_elements) override;

    // binary methods
    void add(dtype* result, dtype* other, size_t num_elements) const override;
    void sub(dtype* result, dtype* other, size_t num_elements) const override;
    void mul(dtype* result, dtype* other, size_t num_elements) const override;
    void div(dtype* result, dtype* other, size_t num_elements) const override;
    void add(dtype* result, dtype scalar, size_t num_elements) const override; // could support Tensor + 1(not a lvalue), (dtype& scalar) can not support this
    void sub(dtype* result, dtype scalar, size_t num_elements) const override;
    void mul(dtype* result, dtype scalar, size_t num_elements) const override;
    void div(dtype* result, dtype scalar, size_t num_elements) const override;
    void pow(dtype* result, dtype scalar, size_t num_elements) const override;

    // reduction methods
    void max(dtype* result, size_t reduce_size, size_t num_elements)    const override;
    void min(dtype* result, size_t reduce_size, size_t num_elements)    const override;
    void sum(dtype* result, size_t reduce_size, size_t num_elements)    const override;
    void argmax(int* result, size_t reduce_size, size_t num_elements) const override;
    void argmin(int* result, size_t reduce_size, size_t num_elements) const override;

// private:
    dtype *data_;

    template <dtype (*op)(dtype, dtype)>
    void applyBinaryOperation(dtype* result, const dtype* other, size_t num_elements) const;
    template <dtype (*op)(dtype, dtype)>
    void applyBinaryScalarOperation(dtype* result,  dtype value, size_t num_elements) const;
    template <dtype (*op)(dtype, dtype)>
    void reduceOperation(dtype* result, size_t reduce_size, size_t num_elements) const;
    template <bool (*comp)(dtype, dtype)>
    void reduceOperationArg(int* result, size_t reduce_size, size_t num_elements) const;
};

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

// copy from CMU10-414 homework project
#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

static CudaVec VecToCuda(const std::vector<int>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

__device__ inline size_t convertIdx(
    size_t linear_index, 
    CudaVec shape,
    CudaVec stride, 
    size_t offset) 
{
    int dim_size = shape.size;
    size_t linear_index_new = 0;
    
    for (int i = dim_size - 1; i >= 0; --i) {
        int cur_dim_id = linear_index % shape.data[i];
        linear_index /= shape.data[i];
        linear_index_new += cur_dim_id * stride.data[i];
    }
    
    return linear_index_new + offset;
}
