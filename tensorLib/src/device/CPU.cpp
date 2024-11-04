#include "Device.hpp"
#include "CPU.hpp"
#include "omp.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>

template class CPU<float>;
template class CPU<int>;

/**
 * matmul operation on CPU
 * @tparam dtype 
 */
template<typename dtype>
void CPU<dtype>::matmul(const dtype* lhs, const dtype* rhs, dtype* result, 
        const std::vector<int>& lhs_stride, 
        const std::vector<int>& rhs_stride, 
        size_t lhs_offset,
        size_t rhs_offset,
        const std::vector<int>& result_shape, 
        size_t result_elements,
        size_t K
        ) 
{
    size_t ndim = result_shape.size();

    #pragma omp parallel for
    for (size_t idx = 0; idx < result_elements; ++idx) {

        size_t linear_index = idx;
        size_t Aoff = lhs_offset, Boff = rhs_offset;

        for (int i = ndim - 1; i >= 0; --i) {
            int cur_dim_id = linear_index % result_shape[i];
            linear_index /= result_shape[i];

            if (i != ndim - 1)
                Aoff += cur_dim_id * lhs_stride[i];
            if (i != ndim - 2)
                Boff += cur_dim_id * rhs_stride[i];
        }

        dtype sum = 0;
        int t1 = lhs_stride[ndim - 1], t2 = rhs_stride[ndim - 2];

        for (int k = 0; k < K; ++k) {
            sum += lhs[Aoff + k * t1] * rhs[Boff + k * t2];
        }

        result[idx] = sum;
    }

}

template <typename dtype>
void CPU<dtype>::full (size_t num_elements, dtype fill_value){
    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
        this->data_[i] = fill_value;
    }
}

template <typename dtype>
dtype CPU<dtype>::getDataLinear(size_t linear_index) const{
    return this->data_[linear_index];
}

template <typename dtype>
void CPU<dtype>::contiguous(
    dtype* result, 
    const std::vector<int>& shape,
    const std::vector<int>& stride, 
    size_t offset,
    size_t num_elements) {

    # pragma omp parallel for
    for (int i=0; i < num_elements; i++) {
        size_t linearIdx = convertIdx(i, shape, stride, offset);
        result[i] = this->data_[linearIdx];
    }
}

template <typename dtype>
void CPU<dtype>::setItemEwise(
    dtype* src,
    const std::vector<int>& shape,
    const std::vector<int>& stride,
    size_t offset,
    size_t num_elements) {

    # pragma omp parallel for
    for (int i=0; i < num_elements; i++) {
        size_t linearIdx = convertIdx(i, shape, stride, offset);
        this->data_[linearIdx] = src[i];
    }
}

template <typename dtype>
void CPU<dtype>::setItemScalar(
    dtype value,
    const std::vector<int>& shape,
    const std::vector<int>& stride,
    size_t offset,
    size_t num_elements) {

    # pragma omp parallel for
    for (int i=0; i < num_elements; i++) {
        size_t linearIdx = convertIdx(i, shape, stride, offset);
        this->data_[linearIdx] = value;
    }
}

////////////////////////////////////////////////////// unary operations ///////////////////////////////////////////////////////////////////////////////
template <typename dtype> inline dtype negFunc(dtype x) { return -x; }
template <typename dtype> inline dtype sinFunc(dtype x) { return std::sin(x); }
template <typename dtype> inline dtype cosFunc(dtype x) { return std::cos(x); }
template <typename dtype> inline dtype expFunc(dtype x) { return std::exp(x); }
template <typename dtype> inline dtype logFunc(dtype x) { return std::log(x); }
template <typename dtype> inline dtype absFunc(dtype x) { return std::abs(x); }
template <typename dtype> inline dtype tanhFunc(dtype x) { return std::tanh(x); }
template <typename dtype> inline dtype siluFunc(dtype x) {
    dtype sigmoid_x = 1 / (1 + std::exp(-x));
    return x * sigmoid_x;
}
template <typename dtype> inline dtype sqrtFunc(dtype x) {
    if (x > 0) {
        return std::sqrt(x);
    } else {
        throw std::domain_error("Cannot take sqrt of non-positive values.");
    }
}
template <typename dtype> inline dtype rsqrtFunc(dtype x) {
    if (x > 0) {
        return 1 / std::sqrt(x);
    } else {
        throw std::domain_error("Cannot take sqrt of non-positive values.");
    }
}

template <typename dtype>
template <dtype (*op)(dtype)>
void CPU<dtype>::applyUnaryOperation(dtype* result, size_t num_elements) const {
    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
        result[i] = op(this->data_[i]);  // Apply function to each element
    }
}

template <typename dtype> void CPU<dtype>::neg(dtype* result, size_t num_elements) { applyUnaryOperation<negFunc<dtype>>(result, num_elements); }
template <typename dtype> void CPU<dtype>::sin(dtype* result, size_t num_elements) { applyUnaryOperation<sinFunc<dtype>>(result, num_elements); }
template <typename dtype> void CPU<dtype>::cos(dtype* result, size_t num_elements) { applyUnaryOperation<cosFunc<dtype>>(result, num_elements); }
template <typename dtype> void CPU<dtype>::exp(dtype* result, size_t num_elements) { applyUnaryOperation<expFunc<dtype>>(result, num_elements); }
template <typename dtype> void CPU<dtype>::log(dtype* result, size_t num_elements) { applyUnaryOperation<logFunc<dtype>>(result, num_elements); }
template <typename dtype> void CPU<dtype>::abs(dtype* result, size_t num_elements) { applyUnaryOperation<absFunc<dtype>>(result, num_elements); }
template <typename dtype> void CPU<dtype>::tanh(dtype* result, size_t num_elements) { applyUnaryOperation<tanhFunc<dtype>>(result, num_elements); }
template <typename dtype> void CPU<dtype>::silu(dtype* result, size_t num_elements) { applyUnaryOperation<siluFunc<dtype>>(result, num_elements); }
template <typename dtype> void CPU<dtype>::sqrt(dtype* result, size_t num_elements) { applyUnaryOperation<sqrtFunc<dtype>>(result, num_elements); }
template <typename dtype> void CPU<dtype>::rsqrt(dtype* result, size_t num_elements) { applyUnaryOperation<rsqrtFunc<dtype>>(result, num_elements); }

////////////////////////////////////////////////////// binary operations ///////////////////////////////////////////////////////////////////////////////
template <typename dtype> static inline dtype addFunc(dtype a, dtype b) { return a + b; }
template <typename dtype> static inline dtype subFunc(dtype a, dtype b) { return a - b; }
template <typename dtype> static inline dtype mulFunc(dtype a, dtype b) { return a * b; }
template <typename dtype> static inline dtype divFunc(dtype a, dtype b) {
    if (b == 0) {
        // or return inf / -inf ?
        throw std::invalid_argument("Division by zero.");
    }
    return a / b;
}
template <typename dtype> static inline dtype powFunc(dtype a, dtype b) { return std::pow(a, b); }

template <typename dtype>
template <dtype (*op)(dtype, dtype)>
void CPU<dtype>::applyBinaryOperation(dtype* result,  const dtype* other, size_t num_elements) const {
    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
        result[i] = op(this->data_[i], other[i]);  // Apply function to each element
    }
}

template <typename dtype> void CPU<dtype>::add(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<addFunc<dtype>>(result, other, num_elements);}
template <typename dtype> void CPU<dtype>::sub(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<subFunc<dtype>>(result, other, num_elements);}
template <typename dtype> void CPU<dtype>::mul(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<mulFunc<dtype>>(result, other, num_elements);}
template <typename dtype> void CPU<dtype>::div(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<divFunc<dtype>>(result, other, num_elements);}

template <typename dtype>
template <dtype (*op)(dtype, dtype)>
void CPU<dtype>::applyBinaryScalarOperation(dtype* result, dtype value, size_t num_elements) const {
    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
        result[i] = op(this->data_[i], value);  // Apply function to each element
    }
}

template <typename dtype> void CPU<dtype>::add(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<addFunc<dtype>>(result, value, num_elements);}
template <typename dtype> void CPU<dtype>::sub(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<subFunc<dtype>>(result, value, num_elements);}
template <typename dtype> void CPU<dtype>::mul(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<mulFunc<dtype>>(result, value, num_elements);}
template <typename dtype> void CPU<dtype>::div(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<divFunc<dtype>>(result, value, num_elements);}
template <typename dtype> void CPU<dtype>::pow(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<powFunc<dtype>>(result, value, num_elements);}
