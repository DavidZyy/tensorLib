#include "../include/Tensor.hpp"
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
        std::shared_ptr<dtype[]> temp(new dtype[num_elements], Deleter<dtype>(num_elements));
        data_ = temp;

        memoryUsage += num_elements * sizeof(dtype);
        std::cout << "Allocate: " << sizeof(dtype) * num_elements << ", now: " << memoryUsage << std::endl;

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
 * Matrix multiplication method implementation
 */
template <typename dtype>
Tensor<dtype> Tensor<dtype>::matmul(const Tensor<dtype>& other) const {
    // Check dimensions for compatibility
    if (shape_.size() != 2 || other.shape().size() != 2 || shape_[1] != other.shape()[0]) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication");
    }

    // make this and other matrix contiguous, which is more efficient when accessing memory for elements.
    auto left = is_contiguous(*this) ? *this : this->contiguous();
    auto right = is_contiguous(other) ? other : other.contiguous();


    // Dimensions of the resulting matrix
    std::vector<int> result_shape = {left.shape_[0], right.shape_[1]};
    Tensor<dtype> result(result_shape);

    // Parallelized matrix multiplication
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < left.shape_[0]; ++i) {
        for (int j = 0; j < right.shape_[1]; ++j) {
            dtype sum = 0;
            for (int k = 0; k < left.shape_[1]; ++k) {
                // sum += left.getData({i, k}) * right.getData({k, j});
                sum += left.data_[i * left.stride_[0] + k * left.stride_[1]] * right.data_[k * right.stride_[0] + j * right.stride_[1]];
            }
            // result.setData({i, j}, sum);
            result.data_[i * result.stride_[0] + j * result.stride_[1]] = sum;
        }
    }

    return result;
}

/**
 * Returns the indices of the maximum values along an axis.
 * @param dim the dimension to reduce.
 */
template <typename dtype>
Tensor<int> Tensor<dtype>::argmax(int dim, bool keepdim) const{
    if (shape_.size() != 2) {
        throw std::invalid_argument("Only support 2d.");
    }

    int reduce_shape = shape_[1 - dim];
    Tensor<int> result(std::vector<int>{reduce_shape});

    int off = stride_[1-dim];
    int stride = stride_[dim];

    for (int i = 0; i < reduce_shape; ++i) {
        int max_index = 0;
        dtype max_value = data_[i*off];
        for (int j = 0; j < shape_[dim]; ++j) {
            if (data_[i*off + j*stride] > max_value) {
                max_value = data_[i*off + j*stride];
                max_index = j;
            }
        }
        result.setData({i}, max_index);
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
 * a recursive method from chatgpt to deal with Tensor of any dimension
 * the select method should be finshed, and should be shallow copy.
 * but maybe not efficient compare to directly mul.
 */

// template <typename dtype>
// Tensor<dtype> Tensor<dtype>::operator*(const Tensor<dtype>& other) const {
//     if (this->shape() != other.shape()) {
//         throw std::invalid_argument("Shapes of tensors are not equal.");
//     }
// 
//     Tensor<dtype> result(this->shape());
//     multiplyRecursive(*this, other, result, 0);
//     return result;
// }
// 
// template <typename dtype>
// void Tensor<dtype>::multiplyRecursive(const Tensor<dtype>& tensor1, const Tensor<dtype>& tensor2, Tensor<dtype>& result, int dim) const {
//     if (dim == tensor1.dimensions() - 1) {
//         // Base case: compute element-wise multiplication for the last dimension
//         for (int i = 0; i < tensor1.shape(dim); i++) {
//             result[i] = tensor1[i] * tensor2[i];
//         }
//     } else {
//         // Recursive case: multiply along the current dimension
//         for (int i = 0; i < tensor1.shape(dim); i++) {
//             multiplyRecursive(tensor1.select(dim, i), tensor2.select(dim, i), result.select(dim, i), dim + 1);
//         }
//     }
// }

/**
 * view use the same data as the original tensor, and reshape copy the data.
 * @tparam dtype 
 */
template <typename dtype>
Tensor<dtype> Tensor<dtype>::view(const std::vector<int>& shape) const {
    if (!is_contiguous(*this)) {
        throw std::invalid_argument("This tensor is not contiguous.");
    }

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
 * only support 4d or 3d Tensor sum up.
 * @tparam dtype 
 */
template<typename dtype>
dtype Tensor<dtype>::sum(bool keepdim) const {
    if (!(shape_.size() == 4 || shape_.size() == 3)) {
        throw std::invalid_argument("Only support 4d.");
    }

    dtype sum = 0;

    if (shape_.size() == 4) {
        for (int i=0; i < this->shape()[0]; i++) {
            for (int j=0; j < this->shape()[1]; j++) {
                for (int k=0; k < this->shape()[2]; k++) {
                    for (int l=0; l < this->shape()[3]; l++) {
                        sum += this->getData({i, j, k, l});
                    }
                }
            }
        }
    } else {
        for (int i=0; i < this->shape()[0]; i++) {
            for (int j=0; j < this->shape()[1]; j++) {
                for (int k=0; k < this->shape()[2]; k++) {
                    sum += this->getData({i, j, k});
                }
            }
        }
    }

    return sum;
}

/**
 * operator * only support 4d and 3d.
 */
template <typename dtype>
Tensor<dtype> Tensor<dtype>::operator*(const Tensor<dtype>& other) const {
    if (!(shape_.size() == 4 || shape_.size() == 3)) {
        throw std::invalid_argument("Only support 4d.");
    }

    if (this->shape() != other.shape()) {
        throw std::invalid_argument("This shape and other shape is not equal.");
    }

    Tensor<dtype> result(this->shape());

    if (shape_.size() == 4) {
        for (int i=0; i < this->shape()[0]; i++) {
            for (int j=0; j < this->shape()[1]; j++) {
                for (int k=0; k < this->shape()[2]; k++) {
                    for (int l=0; l < this->shape()[3]; l++) {
                        result.setData({i, j, k, l}, this->getData({i, j, k, l}) * other.getData({i, j, k, l}));
                    }
                }
            }
        }
    } else {
        for (int i=0; i < this->shape()[0]; i++) {
            for (int j=0; j < this->shape()[1]; j++) {
                for (int k=0; k < this->shape()[2]; k++) {
                        result.setData({i, j, k}, this->getData({i, j, k}) * other.getData({i, j, k}));
                }
            }
        }
    }

    std::cout << "the address of result: " << &result << std::endl;
    std::cout << "the data address of result: " << result.data() << std::endl;
    // return std::move(result);
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
    if (this->shape_.size() != 2) {
        throw std::invalid_argument("only support 2d tensor now");
    }

    Tensor<dtype> result(this->shape());

    for (int i=0; i < this->shape()[0]; i++) {
        for (int j=0; j < this->shape()[1]; j++) {
            result.setData({i, j}, this->getData({i, j}));
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
    // get item first, the new tensor shared the same data with the original tensor in memory.
    slices = process_slices(slices);

    auto out = getItem(slices);

    if (out.shape() != value.shape()) {
        throw std::invalid_argument("The shape of value must be equal to the shape of the slice.");
    }
    
    // current index of out tensor to set value
    std::vector<int> cur_idx(value.shape().size(), 0);
    int num = value.num_elements;

    int cnt = 0;
    // this algorithm maybe not efficient, because it can not use omp to parallelize the loop
    for (int i=0; i < num; i++) {
        int idx = out.offset_;
        // int idx = 0;

        // #pragma omp parallel for // seems error
        for (int j=0; j < cur_idx.size(); j++) {
            idx += cur_idx[j] * out.stride_[j];
        }

        out.data_[idx] = value.data_[cnt++];

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
    if (new_shape.size() != this->shape().size()) {
        throw std::invalid_argument("The new shape must be equal to the original shape.");
    }

    std::vector<int> new_stride;
    for (int i=0; i < new_shape.size(); i++) {
        if (new_shape[i] != this->shape()[i] && this->shape()[i] != 1) {
            throw std::invalid_argument("The dimension to be broadcasted must be 1.");
        }
        new_stride.push_back(this->shape()[i] == 1 ? 0 : this->stride_[i]);
    }

    return Tensor<dtype>(std::move(new_shape), std::move(new_stride), this->offset_, this->data_);
}
