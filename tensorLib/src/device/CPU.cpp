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

template <typename dtype>
void CPU<dtype>::neg(dtype* result, size_t num_elements) {
    applyUnaryOperation(result,
        num_elements,
        [](dtype x) -> dtype{ return -x; }
    );
}

template <typename dtype>
void CPU<dtype>::sin(dtype* result, size_t num_elements) {
    applyUnaryOperation(result,
        num_elements,
        [](dtype x) -> dtype{ return std::sin(x); }
    );
}

template <typename dtype>
void CPU<dtype>::cos(dtype* result, size_t num_elements) {
    applyUnaryOperation(result,
        num_elements,
        [](dtype x) -> dtype{ return std::cos(x); }
    );
}

template <typename dtype>
void CPU<dtype>::exp(dtype* result, size_t num_elements) {
    applyUnaryOperation(result,
        num_elements,
        [](dtype x) -> dtype{ return std::exp(x); }
    );
}

template <typename dtype>
void CPU<dtype>::log(dtype* result, size_t num_elements) {
    applyUnaryOperation(result,
        num_elements,
        [](dtype x) -> dtype{ return std::log(x); }
    );
}

template <typename dtype>
void CPU<dtype>::abs(dtype* result, size_t num_elements) {
    applyUnaryOperation(result,
        num_elements,
        [](dtype x) -> dtype{ return std::abs(x); }
    );
}

template <typename dtype>
void CPU<dtype>::tanh(dtype* result, size_t num_elements) {
    applyUnaryOperation(result,
        num_elements,
        [](dtype x) -> dtype{ return std::tanh(x); }
    );
}

template <typename dtype>
void CPU<dtype>::silu(dtype* result, size_t num_elements) {
    applyUnaryOperation(result,
        num_elements,
        [](dtype x) -> dtype {
            dtype sigmoid_x = 1 / (1 + std::exp(-x));
            return x * sigmoid_x;
        }
    );
}

template <typename dtype>
void CPU<dtype>::sqrt(dtype* result, size_t num_elements) {
    applyUnaryOperation(result,
        num_elements,
        [](dtype x) -> dtype {
            if (x > 0) {
                return std::sqrt(x);
            } else {
                throw std::domain_error("Cannot take sqrt of non-positive values.");
            }
        }
    );
}

template <typename dtype>
void CPU<dtype>::rsqrt(dtype* result, size_t num_elements) {
    applyUnaryOperation(result,
        num_elements,
        [](dtype x) -> dtype {
            if (x > 0) {
                return 1 / std::sqrt(x); // Rsqrt calculation
            } else {
                throw std::domain_error("Cannot take rsqrt of non-positive values.");
            }
        }
    );
}

