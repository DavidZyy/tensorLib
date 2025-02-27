#include "device/Device.hpp"
#include "device/cpu/CPU.hpp"
#include "omp.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <random>

template class CPU<int8_t>;
template class CPU<half>;
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

//         dtype sum = 0;
//         int t1 = lhs_stride[ndim - 1], t2 = rhs_stride[ndim - 2];
// 
//         for (int k = 0; k < K; ++k) {
//             sum += lhs[Aoff + k * t1] * rhs[Boff + k * t2];
//         }
// 
//         result[idx] = sum;

        // Use float for the summation
        float sum = 0.0f;
        int t1 = lhs_stride[ndim - 1], t2 = rhs_stride[ndim - 2];

        for (int k = 0; k < K; ++k) {
            sum += static_cast<float>(lhs[Aoff + k * t1]) * static_cast<float>(rhs[Boff + k * t2]);
        }

        // Convert sum back to dtype (e.g., __half) if necessary
        result[idx] = static_cast<dtype>(sum); // Cast sum (float) to dtype (__half)
    }

}

/**
 * 2D matrix multiplication
 * naive implementation
 * lhs is row major, rhs is col major
 * @tparam dtype 
 */
template <typename dtype>
void CPU<dtype>::matmul2d(const dtype* A, const dtype* B, dtype* C, size_t M, size_t N, size_t K) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // sum += static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * N + j]);
                sum += static_cast<float>(A[i * K + k]) * static_cast<float>(B[j * K + k]);
            }
            C[i * N + j] = static_cast<dtype>(sum);
        }
    }
}

template <typename dtype>
void CPU<dtype>::full (size_t num_elements, dtype fill_value){
    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
        this->data_[i] = fill_value;
    }
}

// Function to create a tensor filled with random numbers from a normal distribution
template <typename dtype>
void CPU<dtype>::randn(size_t num_elements) {
    if constexpr (std::is_integral<dtype>::value) {
        throw std::invalid_argument("randn() is only supported for floating point types.");
    }

    // Set up random number generation for normal distribution
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<float> distribution(0.0, 1.0); // mean = 0.0, stddev = 1.0

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
        this->data_[i] = distribution(generator);
    }
}

template <typename dtype>
dtype CPU<dtype>::getDataLinear(size_t linear_index) const{
    return this->data_[linear_index];
}

template <typename dtype>
void CPU<dtype>::setDataLinear(size_t linear_index, dtype value) {
    this->data_[linear_index] = value;
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
template <typename dtype> inline dtype sinFunc(dtype x) { 
    if constexpr (std::is_same_v<dtype, half>) {
        return __float2half(std::sin(__half2float(x)));
    } else {
        return std::sin(x); 
    }
}
template <typename dtype> inline dtype cosFunc(dtype x) { 
    if constexpr (std::is_same_v<dtype, half>) {
        return __float2half(std::cos(__half2float(x)));
    } else {
        return std::cos(x); 
    }
}
template <typename dtype> inline dtype expFunc(dtype x) { 
    if constexpr (std::is_same_v<dtype, half>) {
        return __float2half(std::exp(__half2float(x)));
    } else {
        return std::exp(x); 
    } 
}
template <typename dtype> inline dtype logFunc(dtype x) { 
    if constexpr (std::is_same_v<dtype, half>) {
        return __float2half(std::log(__half2float(x)));
    } else {
        return std::log(x); 
    }
}
template <typename dtype> inline dtype absFunc(dtype x) {
    if constexpr (std::is_same_v<dtype, half>) {
        return __float2half(std::abs(__half2float(x)));
    } else {
        return std::abs(x);
    }
}
template <typename dtype> inline dtype tanhFunc(dtype x) {
    if constexpr (std::is_same_v<dtype, half>) {
        return __float2half(std::tanh(__half2float(x)));
    } else {
        return std::tanh(x);
    }
}
template <typename dtype> inline dtype siluFunc(dtype x) {
    if constexpr (std::is_same_v<dtype, half>) {
        half sigmoid_x = __float2half(1 / (1 + std::exp(__half2float(x))));
        return x * sigmoid_x;
    } else {
        dtype sigmoid_x = 1 / (1 + std::exp(-x));
        return x * sigmoid_x;
    }
}
template <typename dtype> inline dtype sqrtFunc(dtype x) {
    if constexpr (std::is_same_v<dtype, half>) {
        if (__half2float(x) > 0.0f) {
            return __float2half(std::sqrt(__half2float(x)));
        } else {
            throw std::domain_error("Cannot take sqrt of non-positive values.");
        }
    } else {
        if (x > 0) {
            return std::sqrt(x);
        } else {
            throw std::domain_error("Cannot take sqrt of non-positive values.");
        }
    }
}
template <typename dtype> inline dtype rsqrtFunc(dtype x) {
    if constexpr (std::is_same_v<dtype, half>) {
        if (__half2float(x) > 0.0f) {
            return __float2half(1 / std::sqrt(__half2float(x)));
        } else {
            throw std::domain_error("Cannot take sqrt of non-positive values.");
        }
    } else {
        if (x > 0) {
            return 1 / std::sqrt(x);
        } else {
            throw std::domain_error("Cannot take sqrt of non-positive values.");
        }
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
    if constexpr (std::is_same_v<dtype, half>) {
        if (__half2float(b) == 0) {
            throw std::invalid_argument("Division by zero.");
        }
        return __float2half(__half2float(a) / __half2float(b));
    } else {
        if (b == 0) {
            // or return inf / -inf ?
            throw std::invalid_argument("Division by zero.");
        }
        return a / b;
    }
}
// template <typename dtype> static inline dtype powFunc(dtype a, dtype b) { return std::pow(a, b); }
template <typename dtype> static inline dtype powFunc(dtype a, dtype b) {
    if constexpr (std::is_same_v<dtype, half>) {
        if (__half2float(a) == 0 && __half2float(b) < 0) {
            throw std::invalid_argument("Cannot take negative power of zero.");
        }
        return __float2half(std::pow(__half2float(a), __half2float(b)));
    } else {
        if (a == 0 && b < 0) {
            throw std::invalid_argument("Cannot take negative power of zero.");
        }
        return std::pow(a, b);
    }
}


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

////////////////////////////////////////////////////// reduce operations ///////////////////////////////////////////////////////////////////////////////
template <typename dtype>
template <dtype (*op)(dtype, dtype)>
void CPU<dtype>::reduceOperation(dtype* result, size_t reduce_size, size_t num_elements) const {

    # pragma omp parallel for
    for (int i = 0; i < num_elements; i += reduce_size) {
        dtype temp = this->data_[i];

        // use parallel reduction algorithm if reduce_size > threshold, such as reduce_size = num_elements (find the max/min value or do sum op on all tensor)
        for (int j = 1; j < reduce_size; j++) {
            temp = op(temp, this->data_[i + j]);
        }
        result[i / reduce_size] = temp;
    }
}

template <typename dtype>
template <bool (*comp)(dtype, dtype)>
void CPU<dtype>::reduceOperationArg(int* result, size_t reduce_size, size_t num_elements) const {

    # pragma omp parallel for
    for (int i = 0; i < num_elements; i += reduce_size) {
        dtype best_value = this->data_[i];
        int best_idx = 0;
        for (int j = 1; j < reduce_size; j++) {
            if (comp(this->data_[i + j], best_value)) {
                best_value = this->data_[i + j];
                best_idx = j;
            }
        }
        result[i / reduce_size] = best_idx;
    }
}

// new reduction algorithm
// template <typename dtype>
// template <bool (*comp)(dtype, dtype)>
// void CPU<dtype>::reduceOperationArg(int* result, size_t reduce_size, size_t num_elements) const {
// }

template <typename dtype> static inline dtype maxFunc(dtype a, dtype b) { return std::max(a, b); }
template <typename dtype> static inline dtype minFunc(dtype a, dtype b) { return std::min(a, b); }
template <typename dtype> static inline dtype sumFunc(dtype a, dtype b) { return a + b; }
template <typename dtype> static inline bool argmaxFunc(dtype a, dtype b) { return a > b; }
template <typename dtype> static inline bool argminFunc(dtype a, dtype b) { return a < b; }

template <typename dtype> void CPU<dtype>::max(dtype* result, size_t reduce_size, size_t num_elements)    const { reduceOperation<maxFunc<dtype>>(result, reduce_size, num_elements); }
template <typename dtype> void CPU<dtype>::min(dtype* result, size_t reduce_size, size_t num_elements)    const { reduceOperation<minFunc<dtype>>(result, reduce_size, num_elements); }
template <typename dtype> void CPU<dtype>::sum(dtype* result, size_t reduce_size, size_t num_elements)    const { reduceOperation<sumFunc<dtype>>(result, reduce_size, num_elements); }
template <typename dtype> void CPU<dtype>::mean(dtype* result, size_t reduce_size, size_t num_elements)    const { 
    reduceOperation<sumFunc<dtype>>(result, reduce_size, num_elements); 
    #pragma omp parallel for
    for (int i = 0; i < num_elements / reduce_size; i++) {
        result[i] /= static_cast<dtype>(reduce_size);
    }
}
template <typename dtype> void CPU<dtype>::argmax(int* result, size_t reduce_size, size_t num_elements) const { reduceOperationArg<argmaxFunc<dtype>>(result, reduce_size, num_elements); }
template <typename dtype> void CPU<dtype>::argmin(int* result, size_t reduce_size, size_t num_elements) const { reduceOperationArg<argminFunc<dtype>>(result, reduce_size, num_elements); }

/**
 * the tensor memory is contiguous,
 * regard input's shape as (B*T*n_heads, head_dim)
 */
// template <typename dtype>
// void CPU<dtype>::apply_rotary_emb(const dtype* input, dtype* result, int start_pos, int H, int W) const {
//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < H; i++) {
//         // int offset = start_pos + i * W;
//         for (int j = 0; j < W; j += 2) {
//             int offset = i * W;
//             // dtype theta = 10000.0f * (static_cast<dtype>(j) / static_cast<dtype>(W));
//             dtype theta = start_pos * 1.0f / std::pow(10000.0f, static_cast<dtype>(j) / static_cast<dtype>(W));
//             dtype cos_theta = std::cos(theta);
//             dtype sin_theta = std::sin(theta);
// 
//             dtype v0 = input[offset + j];
//             dtype v1 = input[offset + j + 1];
// 
//             dtype rotary_emb_real = v0 * cos_theta - v1 * sin_theta;
//             dtype rotary_emb_imag = v0 * sin_theta + v1 * cos_theta;
// 
//             result[offset + j] = rotary_emb_real;
//             result[offset + j + 1] = rotary_emb_imag; 
//         }
//     }
// }

template <typename dtype>
void CPU<dtype>::apply_rotary_emb(const dtype* input, dtype* result, int start_pos, int B, int T, int n_heads, int head_dim) const {
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_heads; h++) {
                for (int d = 0; d < head_dim; d += 2) {
                    int offset = b * T * n_heads * head_dim + t * n_heads * head_dim + h * head_dim + d;
                    // dtype theta = (start_pos + t) * 1.0f / std::pow(10000.0f, static_cast<dtype>(d) / static_cast<dtype>(head_dim));
                    // dtype theta;
                    // if constexpr (std::is_same_v<dtype, half>) {
                    //     // theta = (start_pos + t) * 1.0f / std::pow(10000.0f, __float2half(static_cast<float>(d)) / __float2half(static_cast<float>(head_dim)));
                    //     theta = (start_pos + t) * 1.0f / std::pow(10000.0f, __half2float(static_cast<float>(d)) / __half2float(static_cast<float>(head_dim)));
                    // } else {
                    //     theta = (start_pos + t) * 1.0f / std::pow(10000.0f, static_cast<dtype>(d) / static_cast<dtype>(head_dim));
                    // }

                    float theta = (start_pos + t) * 1.0f / std::pow(10000.0f, static_cast<float>(d) / static_cast<float>(head_dim));

                    dtype cos_theta = static_cast<dtype>(std::cos(theta));
                    dtype sin_theta = static_cast<dtype>(std::sin(theta));

                    dtype v0 = input[offset];
                    dtype v1 = input[offset + 1];

                    dtype rotary_emb_real = v0 * cos_theta - v1 * sin_theta;
                    dtype rotary_emb_imag = v0 * sin_theta + v1 * cos_theta;

                    result[offset] = rotary_emb_real;
                    result[offset + 1] = rotary_emb_imag;
                }
            }
        }
    }
}

template <typename dtype>
template <typename OtherType>
void CPU<dtype>::type_cast(dtype* result, const OtherType* src, size_t num_elements) {
    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        if constexpr (std::is_same_v<dtype, half>) {
            result[i] = __float2half(static_cast<float>(src[i]));
        } else {
            result[i] = static_cast<dtype>(src[i]);
        }
    }
}
// Explicit instantiation of the template function for specific types
template void CPU<float>::type_cast<float>(float*, const float*, size_t);
template void CPU<float>::type_cast<int>(float*, const int*, size_t);
template void CPU<float>::type_cast<half>(float*, const half*, size_t);

template void CPU<int>::type_cast<float>(int*, const float*, size_t);
template void CPU<int>::type_cast<int>(int*, const int*, size_t);
template void CPU<int>::type_cast<half>(int*, const half*, size_t);

template void CPU<half>::type_cast<float>(half*, const float*, size_t);
template void CPU<half>::type_cast<int>(half*, const int*, size_t);
template void CPU<half>::type_cast<half>(half*, const half*, size_t);
