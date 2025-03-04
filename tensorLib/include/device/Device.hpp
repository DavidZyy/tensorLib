#pragma once
#include <cstddef>
#include <vector>

template <typename dtype>
class Device {
public:
    Device() : size(0) {}
    Device(size_t num_elements) : size(num_elements) {}
    virtual ~Device() = default;

    // batched matmul
    virtual void matmul(const dtype* lhs, const dtype* rhs, dtype* result, 
        const std::vector<int>& lhs_stride, 
        const std::vector<int>& rhs_stride, 
        size_t lhs_offset,
        size_t rhs_offset,
        const std::vector<int>& result_shape,
        size_t result_elements,
        size_t K) = 0;
    
    virtual void matmul2d(const dtype* A, const dtype* B, dtype* C, size_t M, size_t N, size_t K) = 0;

    virtual dtype* getDataPtr() = 0;  // Pure virtual getter for data_
    virtual void full (size_t num_elements, dtype fill_value) = 0;
    virtual void randn (size_t num_elements) = 0;
    // get data by linear index
    virtual dtype getDataLinear(size_t liner_index) const = 0;
    virtual void setDataLinear(size_t liner_index, dtype value) = 0;
    virtual void contiguous(
        dtype* result, 
        const std::vector<int>& shape,
        const std::vector<int>& stride, 
        size_t offset,
        size_t num_elements) = 0;
    virtual void setItemEwise(
        dtype* src,
        const std::vector<int>& shape,
        const std::vector<int>& stride,
        size_t offset,
        size_t num_elements) = 0;
    virtual void setItemScalar(
        dtype value,
        const std::vector<int>& shape,
        const std::vector<int>& stride,
        size_t offset,
        size_t num_elements) = 0;

    // unary operations
    virtual void neg(dtype* result, size_t num_elements) = 0; // maybe should make these methods const like below
    virtual void sin(dtype* result, size_t num_elements) = 0;
    virtual void cos(dtype* result, size_t num_elements) = 0;
    virtual void exp(dtype* result, size_t num_elements) = 0;
    virtual void log(dtype* result, size_t num_elements) = 0;
    virtual void abs(dtype* result, size_t num_elements) = 0;
    virtual void tanh(dtype* result, size_t num_elements) = 0;
    virtual void silu(dtype* result, size_t num_elements) = 0;
    virtual void sqrt(dtype* result, size_t num_elements) = 0;
    virtual void rsqrt(dtype* result, size_t num_elements) = 0;

    // binary methods
    virtual void add(dtype* result, dtype* other, size_t num_elements) const = 0;
    virtual void sub(dtype* result, dtype* other, size_t num_elements) const = 0;
    virtual void mul(dtype* result, dtype* other, size_t num_elements) const = 0;
    virtual void div(dtype* result, dtype* other, size_t num_elements) const = 0;
    virtual void add(dtype* result, dtype scalar, size_t num_elements) const = 0; // could support Tensor + 1(not a lvalue), (dtype& scalar) can not support this
    virtual void sub(dtype* result, dtype scalar, size_t num_elements) const = 0;
    virtual void mul(dtype* result, dtype scalar, size_t num_elements) const = 0;
    virtual void div(dtype* result, dtype scalar, size_t num_elements) const = 0;
    virtual void pow(dtype* result, dtype scalar, size_t num_elements) const = 0;

    // reduction methods
    virtual void max(dtype* result, size_t reduce_size, size_t num_elements)  const = 0;
    virtual void min(dtype* result, size_t reduce_size, size_t num_elements)  const = 0;
    virtual void sum(dtype* result, size_t reduce_size, size_t num_elements)  const = 0;
    virtual void mean(dtype* result, size_t reduce_size, size_t num_elements)  const = 0;
    virtual void argmax(int* result, size_t reduce_size, size_t num_elements) const = 0;
    virtual void argmin(int* result, size_t reduce_size, size_t num_elements) const = 0;
    virtual void softmax(dtype* output, size_t rows, size_t cols) const = 0;

    // quantization methods
    // virtual void quantize0(dtype* result, size_t num_elements, dtype scale, int zero_point) const = 0;
    // virtual void quantize1(dtype* result, size_t num_elements, dtype scale, int zero_point) const = 0;
    // virtual void dequantize(dtype* result, size_t num_elements, dtype scale, int zero_point) const = 0;

    // special methods
    // virtual void apply_rotary_emb(
    //     const dtype* input,
    //     dtype* result,
    //     int start_pos,
    //     int H,
    //     int W) const = 0;

    virtual void apply_rotary_emb(
        const dtype* input,
        dtype* result,
        int start_pos,
        int B,
        int T,
        int n_heads,
        int head_dim) const = 0;
    
    // do not declare this method in the base class, for template and polymorphism(virtual) can not appear in the same function
    // OtherType -> dtype cast
    // template <typename OtherType>
    // virtual void type_cast(dtype* result, OtherType* src, size_t num_elements) const;

// private:
    size_t size; // number of elements, the total bytes of data_ is: size * sizeof(dtype)
};

