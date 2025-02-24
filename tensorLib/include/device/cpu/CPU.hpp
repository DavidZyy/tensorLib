#pragma once

#include "device/Device.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
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

    ~CPU() override { 
        if (data_ != nullptr)
            delete[] data_; 
    }

    // batched matmul
    void matmul(const dtype* lhs, const dtype* rhs, dtype* result, 
        const std::vector<int>& lhs_stride, 
        const std::vector<int>& rhs_stride, 
        size_t lhs_offset,
        size_t rhs_offset,
        const std::vector<int>& result_shape,
        size_t result_elements,
        size_t K) override;

    void matmul2d(const dtype* A, const dtype* B, dtype* C, size_t M, size_t N, size_t K) override;

    dtype* getDataPtr() override { return data_; }
    void full (size_t num_elements, dtype fill_value) override;
    void randn (size_t num_elements) override;
    dtype getDataLinear(size_t liner_index) const override;
    void setDataLinear(size_t liner_index, dtype value) override;
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
    void neg(dtype* result, size_t num_elements)   override;
    void sin(dtype* result, size_t num_elements)   override;
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
    void mean(dtype* result, size_t reduce_size, size_t num_elements)    const override;
    void argmax(int* result, size_t reduce_size, size_t num_elements) const override;
    void argmin(int* result, size_t reduce_size, size_t num_elements) const override;

    // quantization methods
    // void quantize0(dtype* result, size_t num_elements, dtype scale, int zero_point) const override; // use 0 as zero point
    // void quantize1(dtype* result, size_t num_elements, dtype scale, int zero_point) const override; // calculate new zero point
    // void dequantize(dtype* result, size_t num_elements, dtype scale, int zero_point) const override;

    // special methods
    // void apply_rotary_emb(
    //     const dtype* input,
    //     dtype* result,
    //     int start_pos,
    //     int H,
    //     int W) const override;

    void apply_rotary_emb(
        const dtype* input,
        dtype* result,
        int start_pos,
        int B,
        int T,
        int n_heads,
        int head_dim) const override;

    template <typename OtherType>
    // void type_cast(dtype* result, OtherType src, size_t num_elements) const; // can not use const, for result is data_ to be changed
    void type_cast(dtype* result, const OtherType* src, size_t num_elements);

// private:
    dtype *data_;

    template <dtype (*op)(dtype)>
    void applyUnaryOperation(dtype* result, size_t num_elements) const;

    template <dtype (*op)(dtype, dtype)>
    void applyBinaryOperation(dtype* result, const dtype* other, size_t num_elements) const;
    template <dtype (*op)(dtype, dtype)>
    void applyBinaryScalarOperation(dtype* result, dtype value, size_t num_elements) const;

    template <dtype (*op)(dtype, dtype)>
    void reduceOperation(dtype* result, size_t reduce_size, size_t num_elements) const;
    template <bool (*comp)(dtype, dtype)>
    void reduceOperationArg(int* result, size_t reduce_size, size_t num_elements) const;
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
