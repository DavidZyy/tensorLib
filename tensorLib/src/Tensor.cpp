#include "../include/Tensor.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include <iomanip>
#include "iostream"
#include "math.h"
#include "omp.h"

// Explicit instantiation for int
template class Tensor<int>;

// Explicit instantiation for double
template class Tensor<double>;

// Explicit instantiation for float
template class Tensor<float>;

template class Tensor<uint8_t>;

template <typename dtype>
Tensor<dtype>::Tensor(const std::vector<int>& shape) : ndim(shape.size()), shape_(shape), offset_(0) {
        num_elements = 1; // even shape is empty, it should have 1 elem, means a scala.
        for (int dim : shape) {
            num_elements *= dim;
        }

        // Allocate memory for data, offset, and stride arrays
        // data_ = std::vector<dtype>(num_elements);

        // data_ = std::make_shared<dtype[]>(num_elements); // cpp 20 or later
        // std::shared_ptr<dtype[]> temp(new dtype[num_elements], Deleter<dtype>(num_elements));
        // data_ = temp;

        data_ = std::shared_ptr<dtype[]>(new dtype[num_elements], Deleter<dtype>(num_elements));

        memoryUsage += num_elements * sizeof(dtype);
        // std::cout << "Allocate: " << sizeof(dtype) * num_elements << ", now: " << memoryUsage << std::endl;

        stride_ = std::vector<int>(ndim);

        // Initialize offset and stride arrays
        if (ndim > 0) {
            stride_[ndim - 1] = 1;
            for (int i = ndim - 2; i >= 0; --i) {
                stride_[i] = stride_[i + 1] * shape_[i + 1];
            }
        }
}

template <typename dtype>
// Tensor<dtype>::Tensor(const std::vector<int>& shape, const std::shared_ptr<dtype[]>&& data) 
//     : ndim(shape.size()), shape_(shape), data_(std::move(data)), offset_(0) {    // use move semantic
Tensor<dtype>::Tensor(const std::vector<int>& shape, const std::shared_ptr<dtype[]>& data)
    : ndim(shape.size()), shape_(shape), data_(data), offset_(0) {
        // Calculate the total number of elements in the tensor
        num_elements = 1;
        for (int dim : shape) {
            num_elements *= dim;
        }

        // Allocate memory for data, offset, and stride arrays
        stride_ = std::vector<int>(ndim);

        // Initialize offset and stride arrays
        if (ndim > 0) {
            stride_[ndim - 1] = 1;
            for (int i = ndim - 2; i >= 0; --i) {
                stride_[i] = stride_[i + 1] * shape_[i + 1];
            }
        }
}

/**
 * use std::move semantic to construct a Tensor with given shape, stride, offset, maybe faster ? 
 * @tparam dtype 
 */
template <typename dtype>
Tensor<dtype>::Tensor(const std::vector<int>&& shape, const std::vector<int> &&stride, const int &offset, const std::shared_ptr<dtype[]>& data):
ndim(shape.size()), shape_(std::move(shape)), stride_(std::move(stride)), offset_(offset), data_(data) {
    this-> num_elements = 1;
    for (int dim : shape) {
        this->num_elements *= dim;
    }
}



template <typename dtype>
Tensor<dtype>::~Tensor() {

}

template <typename dtype>
size_t Tensor<dtype>::calculateLinearIndex(const std::vector<int>& indices) const{
    // doulble check
    if (indices.size() != shape_.size() || indices.size() != ndim) {
        throw std::invalid_argument("Error: Indices size does not match tensor dimension");
    }

    size_t linear_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range("Error: Index out of range");
        }
        linear_index += indices[i] * stride_[i];
    }

    return linear_index + offset_;
}

/**
 * maybe should return a Tensor wrapped the data, which is done by pytorch.
 * @tparam dtype 
 */
template <typename dtype>
const dtype& Tensor<dtype>::getData(const std::vector<int>& indices) const {
    size_t linear_index = calculateLinearIndex(indices);

    return data_[linear_index];
}


// Implementation of setData method
template <typename dtype>
void Tensor<dtype>::setData(const std::vector<int>& indices, const dtype& value) {
    size_t linear_index = calculateLinearIndex(indices);

    data_[linear_index] = value;
}

// Accessor implementation (non-const version)
template <typename dtype>
dtype& Tensor<dtype>::operator()(const std::vector<int>& indices) {
    // Calculate linear index from multi-dimensional indices
    size_t linear_index = calculateLinearIndex(indices);
    
    return data_[linear_index];
}

template <typename dtype>
const dtype& Tensor<dtype>::operator()(const std::vector<int>& indices) const {
    // Calculate linear index from multi-dimensional indices
    size_t linear_index = calculateLinearIndex(indices);
    
    return data_[linear_index];
}

template <typename dtype>
void Tensor<dtype>::printTensor(std::ostream& os, size_t depth, std::vector<int> indices) const {
    if (depth == ndim - 1) {
        os << "[";
        auto idx = 0;
        for (auto& dim: indices)
            idx += dim;

        for (int i = 0; i < shape_[depth]; ++i) {
            if (i > 0) os << ", ";
            // os << std::setw(3) << data_[idx + i];
            // os << std::setw(3) << data_[idx + i + offset_];
            os << std::setw(3) << data_[idx + i*stride_[depth] + offset_];
        }
        os << "]";
    } else {
        os << "[";
        for (int i = 0; i < shape_[depth]; ++i) {
            if (i > 0) {
                for (auto i=0; i<ndim-1-depth; i++)
                    os << std::endl;
                for (auto i=0; i<depth+1; i++)
                    os << " ";
            }
            // os << std::endl << " ";
            indices.push_back(i * stride_[depth]);
            printTensor(os, depth + 1, indices);
            if(i != shape_[depth]-1) {
                os << ",";
            }
            indices.pop_back();
        }
        os << "]";
    }
}

/**
 * batched matrix multiplication 
 * @tparam dtype 
 */
template <typename dtype>
// Tensor<dtype> Tensor<dtype>::batched_matmul(const Tensor<dtype>& other) const {
Tensor<dtype> Tensor<dtype>::matmul(const Tensor<dtype>& other) const {
    // Ensure dimensionality is compatible for matrix multiplication
    if (this->ndim < 2 || other.ndim < 2) {
        throw std::invalid_argument("Tensors must have at least 2 dimensions for matmul.");
    }

    // Get the shapes of the tensors
    const auto& A_shape = this->shape_;
    const auto& B_shape = other.shape_;

    // The last dimension of A should match the second-to-last dimension of B
    if (A_shape[this->ndim - 1] != B_shape[other.ndim - 2]) {
        throw std::invalid_argument("Shape mismatch: the number of columns in the first tensor must match the number of rows in the second tensor.");
    }

    // Determine output shape (handling broadcasting for batch dimensions)
    std::vector<int> output_shape;
    size_t num_batch_dims = std::max(this->ndim - 2, other.ndim - 2);

    // broadcast for batched matmul
    Tensor<dtype> A = *this;
    Tensor<dtype> B = other;

    // for example, A.shape = (2, 3, 4, 5), B.shape = (3, 5, 4)
    if (A.shape().size() > B.shape().size()) {
        int diff = A.shape().size() - B.shape().size();
        // check if can be batched
        for (int i = 0; i < B.shape().size()-2; ++i) {
            // ERROR!! SHOULD CHECK BEFORE BROADCAST!!
            if (A.shape()[i+diff] != B.shape()[i]) {
                throw std::invalid_argument("Shape mismatch: the batch dimensions must be broadcastable.");
            }
        }
        // for example, A.shape = (2, 2, 3, 4, 5), B.shape = (3, 5, 4), after this, B.shape will be (1, 1, 3, 5, 4) -> (2, 2, 3, 5, 4)
        std::vector<int> B_new_shape(A.shape().size(), 1);
        for (int i = 0; i < B.shape().size(); ++i) {
            B_new_shape[i+diff] = B.shape()[i];
        }
        B = B.contiguous();
        B = B.view(B_new_shape);
        for (int i = 0; i < diff; ++i) B_new_shape[i] = B.shape()[i];
        B = B.broadcast_to(B_new_shape);
    } else if (A.shape().size() < B.shape().size()) {
        int diff = B.shape().size() - A.shape().size();
        // check if can be batched
        for (int i = 0; i < A.shape().size()-2; ++i) {
            // ERROR!! SHOULD CHECK BEFORE BROADCAST!!
            if (B.shape()[i+diff] != A.shape()[i]) {
                throw std::invalid_argument("Shape mismatch: the batch dimensions must be broadcastable.");
            }
        }
        std::vector<int> A_new_shape(B.shape().size(), 1);
        for (int i = 0; i < A.shape().size(); ++i) {
            A_new_shape[i+diff] = A.shape()[i];
        }
        A = A.contiguous();
        A = A.view(A_new_shape);
        for (int i = 0; i < diff; ++i) A_new_shape[i] = A.shape()[i];
        A = A.broadcast_to(A_new_shape);
    }

    // A.shape()[0: num_batch_dim] == B.shape[0: num_batch_dim] now
    for (int i=0; i<num_batch_dims; ++i) {
        output_shape.push_back(A.shape()[i]);
    }
    output_shape.push_back(A.shape()[A.ndim - 2]);
    output_shape.push_back(B.shape()[B.ndim - 1]);

    // now execute batched matmul
    // Create the result tensor
    Tensor<dtype> result(output_shape);

    std::vector<int> result_indices(output_shape.size(), 0);
    size_t result_elements = result.num_elements;
    for (size_t idx = 0; idx < result_elements; ++idx) {
        std::vector<int> A_indices = result_indices;
        std::vector<int> B_indices = result_indices;

        int row = result_indices[num_batch_dims];
        int col = result_indices[num_batch_dims + 1];
        dtype sum = 0;
        for (int k = 0; k < A.shape()[A.ndim - 1]; ++k) {
            A_indices[A.ndim - 2] = row;
            A_indices[A.ndim - 1] = k;
            B_indices[B.ndim - 2] = k;
            B_indices[B.ndim - 1]= col;
            sum += A.getData(A_indices) * B.getData(B_indices);
        }

        // Store the result in the output tensor
        result.setData(result_indices, sum);
        // Update the result_indices for the next iteration
        for (int dim = output_shape.size() - 1; dim >= 0; --dim) {
            result_indices[dim]++;
            if (result_indices[dim] < output_shape[dim]) {
                break;
            }
            result_indices[dim] = 0;
        }
    }

    return result;
}

/**
 * can not just compare this->data_ and other.data_, because this just means the data_
 * in physical is equal, not the logical.
 * @tparam dtype 
 */
template <typename dtype>
Tensor<int> Tensor<dtype>::operator==(const Tensor<dtype>& other) const {
    if (this->shape() != other.shape()) {
        throw std::invalid_argument("This shape and other shape is not equal.");
    }

    assert(shape_.size() == 1);

    Tensor<int> result(this->shape());

    for (int i = 0; i < shape_[0]; i++) {
        if (this->data_[i] == other.data_[i]) {
            result.setData({i}, 1);
        } else {
            result.setData({i}, 0);
        }
    }

    return result;
}

/**
 * view use the same data as the original tensor, and reshape copy the data.
 * @tparam dtype 
 */
template <typename dtype>
Tensor<dtype> Tensor<dtype>::view(const std::vector<int>& shape) const {
    if (shape == this->shape()) 
        return *this;

    if (!is_contiguous(*this))
        throw std::invalid_argument("This tensor is not contiguous.");

    int num = 1;
    for (auto i=0; i<shape.size(); i++) {
        num *= shape[i];
    }
    if (num != this->num_elements) {
        throw std::invalid_argument("The number of elements is not equal.");
    }

    /**
     * but it seems that this constructor execute a deep copy, not a shadow copy. 
     * it maybe optimized it later.
     */
    Tensor<dtype> result(shape, this->data());
    // Tensor<dtype> result(shape, std::move(this->data()));  // maybe result should take over ownership of data_.

    return result;
}

/**
 * startIdx <= idx < endIdx, not modify the dimension of the tensor.
 * if endIdx = startIdx+1, slice method's behavior is like select, but it not reduce the dimension,
 * the dimension will be 1.
 * 
 * for tesor1[:, :, 0:1] in python, 
 * it can be expressed:  tensor1.slice(0, 1, 2)
 * */ 
template <typename dtype>
Tensor<dtype> Tensor<dtype>::slice(int startIdx, int endIdx, int dim) const {
    if (dim >= this->ndim) {
        throw std::invalid_argument("Dimension out of range.");
    }

    if (startIdx < 0 || endIdx > this->shape_[dim] || startIdx > endIdx) {
        throw std::invalid_argument("Invalid slice range.");
    }

    // copy
    Tensor<dtype> result = *this;
    result.shape_[dim] = endIdx - startIdx;
    result.num_elements = result.num_elements / this->shape_[dim] * result.shape_[dim];

    result.offset_ = this->offset_ + startIdx * this->stride_[dim];

    return result;
}

/**
 * the dimension will be removed, and the total dimension will be reduce 1.
 * @tparam dtype 
 */
template <typename dtype>
Tensor<dtype> Tensor<dtype>::select(int dim, int index) const {
    if (dim >= this->ndim) {
        throw std::invalid_argument("Dimension out of range.");
    }

    if (index >= this->shape_[dim]) {
        throw std::invalid_argument("Invalid slice range.");
    }

    // one dimension is removed
    std::vector<int> new_shape(this->shape().size()-1);
    std::vector<int> new_stride(this->shape().size()-1);

    for (int i=0; i < new_shape.size(); i++) {
        if (i < dim) {
            new_shape[i] = this->shape_[i];
            new_stride[i] = this->stride_[i];
        } else {
            new_shape[i] = this->shape_[i+1];
            new_stride[i] = this->stride_[i+1];
        }
    }

    Tensor<dtype> result(new_shape);
    result.data_ = this->data_;
    result.offset_ = this->offset_ + this->stride_[dim] * index;
    result.stride_ = new_stride;

    // std::cout<<"result data address: "<<&result.data_[0]<<" this data address: "<<&this->data_[0]<<std::endl;
    return result;
}

template <typename dtype>
Tensor<dtype> Tensor<dtype>::transpose(int dim0, int dim1) const {
    Tensor<dtype> result = *this;

    std::swap(result.shape_[dim0], result.shape_[dim1]);
    std::swap(result.stride_[dim0], result.stride_[dim1]);

    return result;
}

template <typename dtype>
Tensor<dtype> Tensor<dtype>::contiguous() const {
    if (is_contiguous(*this)) {
        return *this;
    }

    Tensor<dtype> result(this->shape());

    std::vector<int> cur_idx(this->shape().size(), 0);

    // maybe can parallel this loop later ...
    for (int i=0; i < this->num_elements; i++) {
        int totalIdx = this->offset_;
        for (int j=cur_idx.size()-1; j >= 0; j--) {
            totalIdx += cur_idx[j] * this->stride_[j];
        }
        
        result.data_[i] = this->data_[totalIdx];

        for (int j=cur_idx.size()-1; j >= 0; j--) {
            cur_idx[j] += 1;

            if (cur_idx[j] < result.shape()[j]) {
                break;
            } else {
                cur_idx[j] = 0;
            }
        }
    }

    return result;
}

/**
 * quantize dtype(in most case is float) to int8_t, but store it in int now 
 * in case of overflow when perform matmul.
 * @tparam dtype 
 */
template <typename dtype>
Tensor<int> Tensor<dtype>::quantize() const {
    Tensor<int> result(this->shape());

    // int8 quantization -127 ~ 127
    dtype Q_MAX = 127.0f;

    // find the max absolute value in the tensor
    dtype wmax = 0.0;
    for (int i=0; i < this->num_elements; i++) {
        dtype val = fabs(this->data_[i]);
        if (val > wmax) {
            wmax = val;
        }
    }

    result.scale = wmax / Q_MAX;

    for (int i=0; i < this->num_elements; i++) {
        result.data_[i] = (int)(this->data_[i] / result.scale);
    }

    return result;
}


template <typename dtype>
Tensor<float> Tensor<dtype>::dequantize() const {
    Tensor<float> result(this->shape());

    for (int i=0; i < this->num_elements; i++) {
        result.data_[i] = this->data_[i] * this->scale;
    }

    return result;
}

template <typename dtype>
Tensor<dtype> Tensor<dtype>::getItem(std::vector<std::vector<int>>& slices) const {
    slices = process_slices(slices);

    if (slices.size() != this->shape().size()) {
        throw std::invalid_argument("The number of slices must be equal to the number of dimensions.");
    }

    std::vector<int> new_shape;
    std::vector<int> new_stride;
    int new_offset = this->offset_;
    
    for (int i=0; i < slices.size(); i++) {
        int start = slices[i][0], stop = slices[i][1], step = slices[i][2];

        new_shape.push_back((stop - start + (step - 1)) / step);
        new_stride.push_back(step * this->stride_[i]);
        new_offset += start * this->stride_[i];
    }

    // Tensor<dtype> result(std::move(new_shape), std::move(new_stride), std::move(new_offset), this->data_);
    Tensor<dtype> result(std::move(new_shape), std::move(new_stride), new_offset, this->data_);
    return result;
}

/**
 * The slices is a vector of vector, each vector represent a slice, have 3 ints, [start, stop, step],
 * step is default to 1 if not provided. start is default to 0 if not provided, stop is default to this->shape[dim] if not provided.
 *
 * value is contiguous, out is not contiguous
 * @tparam dtype 
 */
template <typename dtype>
void Tensor<dtype>::setItem(std::vector<std::vector<int>>& slices, const Tensor<dtype>& value) {
    // value = value.contiguous();
    if (!is_contiguous(value)) {
        throw std::invalid_argument("The value must be contiguous.");
    }

    // get item first, the new tensor shared the same data with the original tensor in memory.
    slices = process_slices(slices);

    auto out = getItem(slices);

    if (out.shape() != value.shape()) {
        throw std::invalid_argument("The shape of value must be equal to the shape of the slice.");
    }
    
    // current index of out tensor to set value
    // std::vector<int> cur_idx(value.shape().size(), 0);
    std::vector<int> cur_idx(out.shape().size(), 0);

    // maybe can parallel this loop later ...
    // for (int i=0; i < value.num_elements; i++) {
    for (int i=0; i < out.num_elements; i++) {
        int idx = out.offset_;

        // #pragma omp parallel for // seems error
        for (int j=0; j < cur_idx.size(); j++) {
            idx += cur_idx[j] * out.stride_[j]; // use current idx to calculate the flatten idx
        }

        out.data_[idx] = value.data_[i]; // index value use i directly, value should be contiguous.

        // carry
        // for (int j=0; j < cur_idx.size(); j++) { // this is not right, because stride[0] is the max stride, stride[dim-1] is the min stride, we should increse the cur_idx from bigger dimension to smaller
        for (int j=cur_idx.size()-1; j >= 0; j--) {
            cur_idx[j] += 1;

            if (cur_idx[j] < out.shape()[j]) {
                break;
            } else {
                cur_idx[j] = 0;
            }
        }
    }
}

/**
 * set item with a scalar value
 * @tparam dtype 
 */
template <typename dtype>
void Tensor<dtype>::setItem(std::vector<std::vector<int>>& slices, dtype value) {
    // get item first, the new tensor shared the same data with the original tensor in memory.
    slices = process_slices(slices);

    auto out = getItem(slices);
    
    // current index of out tensor to set value
    // std::vector<int> cur_idx(value.shape().size(), 0);
    std::vector<int> cur_idx(out.shape().size(), 0);

    // maybe can parallel this loop later ...
    // for (int i=0; i < value.num_elements; i++) {
    for (int i=0; i < out.num_elements; i++) {
        int idx = out.offset_;

        // #pragma omp parallel for // seems error
        for (int j=0; j < cur_idx.size(); j++) {
            idx += cur_idx[j] * out.stride_[j]; // use current idx to calculate the flatten idx
        }

        out.data_[idx] = value;

        // carry
        // for (int j=0; j < cur_idx.size(); j++) { // this is not right, because stride[0] is the max stride, stride[dim-1] is the min stride, we should increse the cur_idx from bigger dimension to smaller
        for (int j=cur_idx.size()-1; j >= 0; j--) {
            cur_idx[j] += 1;

            if (cur_idx[j] < out.shape()[j]) {
                break;
            } else {
                cur_idx[j] = 0;
            }
        }
    }
}

/**
 * The format of slice is [start, stop, step], default step is 1.
 * process_slices will convert the slice to the format like [start, stop, step].
 * @tparam dtype 
 */
template<typename dtype>
std::vector<std::vector<int>> Tensor<dtype>::process_slices(const std::vector<std::vector<int>>& slices) const {
    std::vector<std::vector<int>> result;

    for (int i=0; i < slices.size(); i++) {
        auto slice = slices[i];

        if (slice.empty()) { // whole
            result.push_back({0, this->shape()[i], 1}); 
        } else if (slice.size() == 1) { // select, only have start
            result.push_back({slice[0], slice[0]+1, 1});
        } else if (slice.size() == 2) { // slice, have start and stop
            result.push_back({slice[0], slice[1], 1});
        } else {
            result.push_back(slice); // slice, have start, stop and step
        } 
    }

    return result;
}

/*
 * Broadcast an array to a new shape.  new_shape's elements must be the
 * same as the original shape, except for dimensions in the self where
 * the size = 1 (which can then be broadcast to any size). 
 * This will not copy memory, and just achieves
 * broadcasting by manipulating the strides.
 * 
 * so when broadcast_to a shape().size() greater than current shape().size(), you should add 1 in current shape()'s dimension which to be broadcasted.
 */
template<typename dtype>
Tensor<dtype> Tensor<dtype>::broadcast_to(const std::vector<int>& new_shape) const {
    if (this->shape() == new_shape) 
        return *this;

    auto prepend_shape = this->shape(); // if this->shape().size() < new_shape().size, prepend 1 before this->shape().

    if (prepend_shape.size() > new_shape.size()) {
        throw std::invalid_argument("The new shape must be greater than or equal to the original shape.");
    } else if (prepend_shape.size() < new_shape.size()) {
        prepend_shape.insert(prepend_shape.begin(), new_shape.size() - prepend_shape.size(), 1);
    }

    auto new_tensor = this->view(prepend_shape);

    // now prepend_shape.size() == new_shape.size()
    std::vector<int> new_stride;
    for (int i=0; i < new_shape.size(); i++) {
        if ((new_shape[i] != prepend_shape[i]) && prepend_shape[i] != 1) {
            throw std::invalid_argument("The dimension to be broadcasted must be 1.");
        }
        new_stride.push_back(prepend_shape[i] == 1 ? 0 : new_tensor.stride_[i]);
    }

    return Tensor<dtype>(std::move(new_shape), std::move(new_stride), new_tensor.offset_, new_tensor.data_);
}

// negative operator
template <typename dtype>
inline Tensor<dtype> Tensor<dtype>::operator-() const {
    return applyUnaryOperation([](dtype x) -> dtype{
        return -x;
    });
}

template <typename dtype>
inline Tensor<dtype> Tensor<dtype>::sin() const {
    return applyUnaryOperation([](dtype x) -> dtype{
        return std::sin(x);
    });
}

template <typename dtype>
inline Tensor<dtype> Tensor<dtype>::cos() const {
    return applyUnaryOperation([](dtype x) -> dtype{
        return std::cos(x);
    });
}

template <typename dtype>
inline Tensor<dtype> Tensor<dtype>::exp() const {
    return applyUnaryOperation([](dtype x) -> dtype{
        return std::exp(x);
    });
}

template <typename dtype>
inline Tensor<dtype> Tensor<dtype>::log() const {
    return applyUnaryOperation([](dtype x) -> dtype{
        return std::log(x);
    });
}

template <typename dtype>
inline Tensor<dtype> Tensor<dtype>::abs() const {
    return applyUnaryOperation([](dtype x) -> dtype{
        return std::fabs(x);
    });
}

template <typename dtype>
inline Tensor<dtype> Tensor<dtype>::tanh() const {
    return applyUnaryOperation([](dtype x) -> dtype{
        return std::tanh(x);
    });
}

// Custom logic with inline for "silu"
template <typename dtype>
inline Tensor<dtype> Tensor<dtype>::silu() const {
    return applyUnaryOperation([](dtype x) -> dtype {
        dtype sigmoid_x = 1 / (1 + std::exp(-x));
        return x * sigmoid_x;
    });
}

template <typename dtype>
inline Tensor<dtype> Tensor<dtype>::sqrt() const {
    return applyUnaryOperation([](dtype x) -> dtype {
        if (x > 0) {
            return std::sqrt(x); // Rsqrt calculation
        } else {
            throw std::domain_error("Cannot take rsqrt of non-positive values.");
        }
    });
}

template <typename dtype>
inline Tensor<dtype> Tensor<dtype>::rsqrt() const {
    return applyUnaryOperation([](dtype x) -> dtype {
        if (x > 0) {
            return 1 / std::sqrt(x); // Rsqrt calculation
        } else {
            throw std::domain_error("Cannot take rsqrt of non-positive values.");
        }
    });
}

template<typename dtype>
Tensor<dtype> Tensor<dtype>::permute(const std::vector<int>& new_axes) const {
    if (new_axes.size() != this->shape().size()) {
        throw std::invalid_argument("The new axes must be equal to the original axes.");
    }

    std::vector<int> new_shape;
    std::vector<int> new_stride;
    for (int i=0; i < new_axes.size(); i++) {
        new_shape.push_back(this->shape_[new_axes[i]]);
        new_stride.push_back(this->stride_[new_axes[i]]);
    }

    return Tensor<dtype>(std::move(new_shape), std::move(new_stride), this->offset_, this->data_);
}

template <typename dtype>
int Tensor<dtype>::handle_axis(int axis) const {
    int dims = static_cast<int>(this->shape().size()); // size is unsigned, so use int
    if (axis >= dims) {
        throw std::invalid_argument("The axis must be less than the shape size.");
    } else if (axis < -dims) {
        throw std::invalid_argument("The axis must be greater than or equal to -shape size.");
    }

    if (axis < 0) {
        axis += this->shape().size();
    }
    return axis;
}

/**
 * permute the axis to the last dimension first, then return a view of the tensor which is contiguous.
 * @tparam dtype 
 */
template<typename dtype>
Tensor<dtype> Tensor<dtype>::get_reduce_view(int axis) const {
    std::vector<int> new_axes;
    for (int i=0; i < this->shape().size(); i++) {
        if (i != axis) {
            new_axes.push_back(i);
        }
    }
    new_axes.push_back(axis);

    auto view = this->permute(new_axes);
    view = view.contiguous();
    return view;
}

template<typename dtype>
std::vector<int> Tensor<dtype>::get_reduce_shape(int axis, bool keepdims) const {
    std::vector<int> new_shape = this->shape();
    if (keepdims) {
        new_shape[axis] = 1;
    } else {
        new_shape.erase(new_shape.begin() + axis);
    }
    
    return new_shape;
}

template<typename dtype>
Tensor<dtype> Tensor<dtype>::reduce(int axis, bool keepdims, dtype(*op)(dtype, dtype)) const {
    // Handle the axis properly, permute to move the axis to reduce to the last dimension
    axis = handle_axis(axis);
    auto view = get_reduce_view(axis);
    auto new_shape = get_reduce_shape(axis, keepdims);

    Tensor<dtype> result(new_shape);

    int reduce_size = this->shape()[axis];
    
    // Apply the operation for each reduced chunk
    for (int i = 0; i < view.num_elements; i += reduce_size) {
        dtype temp = view.data_[i];  // Initialize the temp value
        for (int j = 1; j < reduce_size; j++) {
            temp = op(temp, view.data_[i + j]);  // Apply the operation
        }
        result.data_[i / reduce_size] = temp;  // Store the result
    }

    return result;
}

template<typename dtype>
Tensor<dtype> Tensor<dtype>::max(int axis, bool keepdims) const {
    return reduce(axis, keepdims, [](dtype a, dtype b) { return std::max(a, b); });
}

template<typename dtype>
Tensor<dtype> Tensor<dtype>::min(int axis, bool keepdims) const {
    return reduce(axis, keepdims, [](dtype a, dtype b) { return std::min(a, b); });
}

template<typename dtype>
Tensor<dtype> Tensor<dtype>::sum(int axis, bool keepdims) const {
    return reduce(axis, keepdims, add<dtype>);
}

template<typename dtype>
Tensor<dtype> Tensor<dtype>::mean(int axis, bool keepdims) const {
    int reduce_size = this->shape()[handle_axis(axis)];

    auto result1 = this->sum(axis, keepdims);
    auto result2 = result1 / static_cast<dtype>(reduce_size);

    return result2;
}

template<typename dtype>
Tensor<int> Tensor<dtype>::reduce_arg(int axis, bool keepdims, bool(*comp)(dtype, dtype)) const {
    // Handle the axis properly, permute to move the axis to reduce to the last dimension
    axis = handle_axis(axis);
    auto view = get_reduce_view(axis);
    auto new_shape = get_reduce_shape(axis, keepdims);

    Tensor<int> result(new_shape);

    int reduce_size = this->shape()[axis];
    
    // Iterate over the tensor in chunks (based on the reduce size)
    for (int i = 0; i < view.num_elements; i += reduce_size) {
        auto best_index = 0;
        auto best_value = view.data_[i];  // Initialize with the first element
        
        // Find the index with the maximum value (or other comparison)
        for (int j = 1; j < reduce_size; j++) {
            if (comp(view.data_[i + j], best_value)) {
                best_value = view.data_[i + j];
                best_index = j;
            }
        }
        result.data_[i / reduce_size] = best_index;  // Store the index of the best value
    }

    return result;
}

template<typename dtype>
Tensor<int> Tensor<dtype>::argmax(int axis, bool keepdims) const {
    return reduce_arg(axis, keepdims, [](dtype a, dtype b) { return a > b; });
}

template<typename dtype>
Tensor<int> Tensor<dtype>::argmin(int axis, bool keepdims) const {
    return reduce_arg(axis, keepdims, [](dtype a, dtype b) { return a < b; });
}

template<typename dtype>
Tensor<dtype> Tensor<dtype>::softmax(int dim) const {
    auto max_val = this->max(dim, true);
    max_val = max_val.broadcast_to(this->shape());

    auto exp_val = (*this - max_val).exp();

    auto sum_val = exp_val.sum(dim, true);
    sum_val = sum_val.broadcast_to(this->shape());

    // std::cout << *this << std::endl;
    // std::cout << max_val << std::endl;
    // std::cout << exp_val << std::endl;
    // std::cout << sum_val << std::endl;
    return exp_val / sum_val;
}

/**
 * used for implicit broadcasting 
 * implicitly broadcasting before operation, for example:
 * a(5, 1, 3) + b(4, 3) -> a(5, 1, 3) + b(1, 4, 3) -> new_shape(5, 4, 3)
 */
template <typename dtype>
std::vector<int> Tensor<dtype>::get_broadcast_shape(std::vector<int>& shape_a, std::vector<int>& shape_b) const {
    if (shape_a == shape_b) return shape_a;

    auto a = shape_a;
    auto b = shape_b;

    int dims_a = shape_a.size();
    int dims_b = shape_b.size();
    int max_dims = std::max(dims_a, dims_b);

    if (dims_a > dims_b) {
        for (int i=dims_b; i < dims_a; i++) {
            b.insert(b.begin(), 1);
        }
    } else {
        for (int i=dims_a; i < dims_b; i++) {
            a.insert(a.begin(), 1);
        }
    }

    // now a.size() == b.size()
    std::vector<int> new_shape;
    for (int i=0; i <max_dims; i++) {
        if (a[i] == b[i]) {
            new_shape.push_back(a[i]);
        } else if (a[i] == 1) {
            new_shape.push_back(b[i]);
        } else if (b[i] == 1) {
            new_shape.push_back(a[i]);
        } else {
            throw std::invalid_argument("The shape cannot be broadcasted.");
        }
    }

    return new_shape;
}

/**
 * maybe should support implicit broadcasting 
 * @tparam dtype 
 */
template <typename dtype>
Tensor<dtype> Tensor<dtype>::applyBinaryOperation(const Tensor<dtype>& other, dtype(*op)(dtype, dtype)) const {
    Tensor<dtype> a = *this, b = other;
    
    // implicit broadcasting
    if (this->shape() != other.shape()) {
        std::vector<int> shape_a = this->shape();
        std::vector<int> shape_b = other.shape();
        auto new_shape = get_broadcast_shape(shape_a, shape_b);
        a = a.broadcast_to(new_shape); 
        b = b.broadcast_to(new_shape);
    }

    a = a.contiguous(); // if broadcast have influence on the result of contiguous?? (seems no, because contiguous will allocate actually memory for array)
    b = b.contiguous();

    Tensor<dtype> result(this->shape());
    #pragma omp parallel for
    for (int i = 0; i < this->num_elements; ++i) {
        result.data_[i] = op(a.data_[i], b.data_[i]);
    }

    return result;
}

/**
 * if this is a broadcasted tensor, need to use contiguous() firtst, or the num_elements is not the actual elem size of the tensor data_.
 * @tparam dtype 
 */
template <typename dtype>
Tensor<dtype> Tensor<dtype>::applyBinaryScalarOperation(dtype scalar, dtype(*op)(dtype, dtype)) const {
    Tensor<dtype> result = this->contiguous();

    #pragma omp parallel for
    for (int i = 0; i < result.num_elements; ++i) { // broadcast may get error, for it does not have num_elements elems actually.
        result.data_[i] = op(result.data_[i], scalar);
    }

    return result;
}

template <typename dtype> Tensor<dtype> Tensor<dtype>::operator+(const Tensor<dtype>& other) const { return applyBinaryOperation(other, add<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator-(const Tensor<dtype>& other) const { return applyBinaryOperation(other, subtract<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator*(const Tensor<dtype>& other) const { return applyBinaryOperation(other, multiply<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator/(const Tensor<dtype>& other) const { return applyBinaryOperation(other, divide<dtype>); }

template <typename dtype> Tensor<dtype> Tensor<dtype>::operator+(dtype scalar) const { return applyBinaryScalarOperation(scalar, add<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator-(dtype scalar) const { return applyBinaryScalarOperation(scalar, subtract<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator*(dtype scalar) const { return applyBinaryScalarOperation(scalar, multiply<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator/(dtype scalar) const { return applyBinaryScalarOperation(scalar, divide<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::pow(dtype scalar) const { return applyBinaryScalarOperation(scalar, power<dtype>); }
