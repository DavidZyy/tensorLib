#pragma once

#include "Device.hpp"
#include <cstddef>
#include <vector>

template <typename dtype>
class CPU : public Device<dtype> {
public:
    // Default constructor
    CPU() : Device<dtype>(0), data_(nullptr) {}
    CPU(size_t size, dtype *ptr) : Device<dtype>(size), data_(ptr) {}
    CPU(size_t size) : Device<dtype>(size) { 
        data_ = new dtype[size]; 
    }

    ~CPU() override { delete[] data_; }

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

// private:
    dtype *data_;

    inline void applyUnaryOperation(dtype *result, size_t num_elements, dtype (*func)(dtype)) {
        #pragma omp parallel for
        for (size_t i = 0; i < num_elements; ++i) {
            result[i] = func(this->data_[i]);  // Apply function to each element
        }
    }
};

inline size_t convertIdx(size_t linear_index, const std::vector<int>& shape, const std::vector<int>& stride, size_t offset) {
    size_t linear_index_new = 0;

    for (int i = shape.size() - 1; i >= 0; --i) {
        int cur_dim_id = linear_index % shape[i];
        linear_index /= shape[i];
        linear_index_new += cur_dim_id * stride[i];
    }

    return linear_index_new + offset;
}
