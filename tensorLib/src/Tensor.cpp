#include "../include/Tensor.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include <iomanip>
#include "Device.hpp"
#include "iostream"
#include "math.h"
#include "omp.h"
#include "CPU.hpp"
#include "CUDA.hpp"

// Explicit instantiation for int
template class Tensor<int>;

// Explicit instantiation for double
// template class Tensor<double>;

// Explicit instantiation for float
template class Tensor<float>;

// template class Tensor<uint8_t>;

template <typename dtype>
Tensor<dtype>::Tensor(const std::vector<int>& shape, const std::string& device_type) : ndim(shape.size()), shape_(shape), offset_(0), device_type(device_type) {
        num_elements = 1; // even shape is empty, it should have 1 elem, means a scala.
        for (int dim : shape) {
            num_elements *= dim;
        }

        this->data_ = std::shared_ptr<dtype[]>(new dtype[num_elements], Deleter<dtype>(num_elements));

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

        if (device_type == "cpu") {
            this->device = std::shared_ptr<CPU<dtype>>(new CPU<dtype>(num_elements));
        } else if (device_type == "cuda") {
            this->device = std::shared_ptr<CUDA<dtype>>(new CUDA<dtype>(num_elements));
        } else {
            throw std::invalid_argument("Invalid device name");
        }
}

template <typename dtype>
Tensor<dtype>::Tensor(const std::vector<int>& shape, const std::shared_ptr<dtype[]>& data, const std::string& device_type)
    : ndim(shape.size()), shape_(shape), data_(data), offset_(0), device_type(device_type) {
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

        
        if (device_type == "cpu") {
            this->device = std::shared_ptr<CPU<dtype>>(new CPU<dtype>(num_elements));
        } else if (device_type == "cuda") {
            this->device = std::shared_ptr<CUDA<dtype>>(new CUDA<dtype>(num_elements));
        } else {
            throw std::invalid_argument("Invalid device name");
        }
}

/**
 * use std::move semantic to construct a Tensor with given shape, stride, offset, maybe faster ? 
 * @tparam dtype 
 */
template <typename dtype>
Tensor<dtype>::Tensor(const std::vector<int>&& shape, const std::vector<int> &&stride, const int &offset, const std::shared_ptr<dtype[]>& data, const std::string& device_type):
ndim(shape.size()), shape_(std::move(shape)), stride_(std::move(stride)), offset_(offset), data_(data), device_type(device_type) {
    this-> num_elements = 1;
    for (int dim : shape) {
        this->num_elements *= dim;
    }

    if (device_type == "cpu") {
        this->device = std::shared_ptr<CPU<dtype>>(new CPU<dtype>(num_elements));
    } else if (device_type == "cuda") {
        this->device = std::shared_ptr<CUDA<dtype>>(new CUDA<dtype>(num_elements));
    } else {
        throw std::invalid_argument("Invalid device name");
    }
}

/**
 * use std::move semantic to construct a Tensor with given shape, stride, offset, maybe faster ?
 * NOTE: this constructor is used in tensor_bindings.convert_to_tensor(), the data pointer data_ptr should on device you pass, and freed by this
 * tensor, or it may cause error!
 * @tparam dtype 
 */
// template <typename dtype>
// Tensor<dtype>::Tensor(const std::vector<int>&& shape, const std::vector<int> &&stride, const int &offset, dtype* data_ptr, const std::string& device_type):
// ndim(shape.size()), shape_(std::move(shape)), stride_(std::move(stride)), offset_(offset), device_type(device_type) {
//     this-> num_elements = 1;
//     for (int dim : shape) {
//         this->num_elements *= dim;
//     }
// 
//     if (device_type == "cpu") {
//         this->device = std::shared_ptr<CPU<dtype>>(new CPU<dtype>(data_ptr));
//     } else if (device_type == "cuda") {
//         this->device = std::shared_ptr<CUDA<dtype>>(new CUDA<dtype>(data_ptr));
//     } else {
//         throw std::invalid_argument("Invalid device name");
//     }
// }

template <typename dtype>
Tensor<dtype>::Tensor(const std::vector<int>& shape, const std::shared_ptr<Device<dtype>>& device, const std::string& device_type) 
: ndim(shape.size()), shape_(shape), offset_(0), device(device), device_type(device_type)
{
        this->num_elements = 1;
        for (int dim : shape) {
            this->num_elements *= dim;
        }

        // Allocate memory for data, offset, and stride arrays
        this->stride_ = std::vector<int>(ndim);

        // Initialize offset and stride arrays
        if (ndim > 0) {
            this->stride_[ndim - 1] = 1;
            for (int i = ndim - 2; i >= 0; --i) {
                this->stride_[i] = this->stride_[i + 1] * this->shape_[i + 1];
            }
        }
}

template <typename dtype>
Tensor<dtype>::Tensor(const std::vector<int>&& shape, const std::vector<int> &&stride, const int &offset, const std::shared_ptr<Device<dtype>>& device, const std::string& device_type):
ndim(shape.size()), shape_(std::move(shape)), stride_(std::move(stride)), offset_(offset), device(device), device_type(device_type) {
    this-> num_elements = 1;
    for (int dim : shape) {
        this->num_elements *= dim;
    }
}

template <typename dtype>
Tensor<dtype>::~Tensor() {

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
            // os << std::setw(3) << data_[idx + i*stride_[depth] + offset_];
            os << std::setw(3) << this->device->getDataLinear(idx + i*stride_[depth] + offset_);
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

template <typename dtype>
Tensor<dtype> Tensor<dtype>::matmul(const Tensor<dtype>& other) const {
    if (this->device_type != other.device_type) {
        throw std::invalid_argument("Tensors must be on the same device.");
    }

    // Ensure dimensionality is compatible for matrix multiplication
    if (this->ndim < 2 || other.ndim < 2) {
        throw std::invalid_argument("Tensors must have at least 2 dimensions for matmul.");
    }

    // The last dimension of A should match the second-to-last dimension of B
    if (this->shape_[this->ndim - 1] != other.shape_[other.ndim - 2]) {
        throw std::invalid_argument("Shape mismatch: the number of columns in the first tensor must match the number of rows in the second tensor.");
    }

    Tensor<dtype> A = *this;
    Tensor<dtype> B = other;

    size_t num_batch_dims = std::max(A.ndim - 2, B.ndim - 2);
    size_t dim_diff = std::abs(static_cast<int>(A.ndim) - static_cast<int>(B.ndim));

    // If needed, prepend dimensions to match larger tensor size
    std::vector<int> A_broadcast_shape = A.shape_;
    std::vector<int> B_broadcast_shape = B.shape_;
    std::vector<int> output_shape;

    // for example, A.shape = (2, 2, 3, 4, 5), B.shape = (3, 5, 4), after this, B.shape will be (1, 1, 3, 5, 4) -> (2, 2, 3, 5, 4)
    if (A.ndim < B.ndim) {
        A_broadcast_shape.insert(A_broadcast_shape.begin(), dim_diff, 1);
    } else if (B.ndim < A.ndim) {
        B_broadcast_shape.insert(B_broadcast_shape.begin(), dim_diff, 1);
    }

    // Adjust batch dimensions to be broadcast-compatible
    for (size_t i = 0; i < num_batch_dims; ++i) {
        if (A_broadcast_shape[i] != B_broadcast_shape[i]) {
            if (A_broadcast_shape[i] == 1) {
                A_broadcast_shape[i] = B_broadcast_shape[i];
            } else if (B_broadcast_shape[i] == 1) {
                B_broadcast_shape[i] = A_broadcast_shape[i];
            } else {
                throw std::invalid_argument("Shape mismatch: the batch dimensions must be broadcastable.");
            }
        }
        output_shape.push_back(A_broadcast_shape[i]);
    }

    A = A.broadcast_to(A_broadcast_shape);
    B = B.broadcast_to(B_broadcast_shape);

    output_shape.push_back(A.shape_[A.ndim - 2]);
    output_shape.push_back(B.shape_[B.ndim - 1]);
    int height = A.shape_[A.ndim - 2];
    int width = B.shape_[B.ndim - 1];
    int K = A.shape_[A.ndim - 1];

    // now execute batched matmul
    Tensor<dtype> result(output_shape, this->device_type); // for get pass device will get bug !!

    size_t result_elements = result.num_elements;

    this->device->matmul(
        this->device->getDataPtr(),
        other.device->getDataPtr(),
        result.device->getDataPtr(),
        A.stride_,
        B.stride_,
        A.offset_,
        B.offset_,
        result.shape_,
        result_elements,
        K);

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

    // Tensor<dtype> result(shape, this->data());
    Tensor<dtype> result(shape, this->device, this->device_type);

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

    std::vector<std::vector<int>> slices;
    for (int i=0; i<this->ndim; i++) {
        if (i == dim) {
            slices.push_back({startIdx, endIdx});
        } else {
            slices.push_back({}); // select whole
        }
    }

    return getItem(slices);
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

    std::vector<std::vector<int>> slices;
    for (int i=0; i<this->ndim; i++) {
        if (i == dim) {
            slices.push_back({index, index+1});
        } else {
            slices.push_back({});
        }
    }
    return getItem(slices);
}

template <typename dtype>
Tensor<dtype> Tensor<dtype>::contiguous() const {
    if (is_contiguous(*this)) {
        return *this;
    }

    Tensor<dtype> result(this->shape(), this->device_type);

    this->device->contiguous(
        result.device->getDataPtr(), 
        this->shape_,
        this->stride_,
        this->offset_,
        this->num_elements);

    return result;
}

/**
 * quantize dtype(in most case is float) to int8_t, but store it in int now 
 * in case of overflow when perform matmul.
 * @tparam dtype 
 */
// template <typename dtype>
// Tensor<int> Tensor<dtype>::quantize() const {
//     Tensor<int> result(this->shape());
// 
//     // int8 quantization -127 ~ 127
//     dtype Q_MAX = 127.0f;
// 
//     // find the max absolute value in the tensor
//     dtype wmax = 0.0;
//     for (int i=0; i < this->num_elements; i++) {
//         dtype val = fabs(this->data_[i]);
//         if (val > wmax) {
//             wmax = val;
//         }
//     }
// 
//     result.scale = wmax / Q_MAX;
// 
//     for (int i=0; i < this->num_elements; i++) {
//         result.data_[i] = (int)(this->data_[i] / result.scale);
//     }
// 
//     return result;
// }
// 
// 
// template <typename dtype>
// Tensor<float> Tensor<dtype>::dequantize() const {
//     Tensor<float> result(this->shape());
// 
//     for (int i=0; i < this->num_elements; i++) {
//         result.data_[i] = this->data_[i] * this->scale;
//     }
// 
//     return result;
// }

template <typename dtype>
Tensor<dtype> Tensor<dtype>::getItem(std::vector<std::vector<int>>& slices) const {
    // assert(this->shape().size() == this->ndim);
    slices = process_slices(slices);

    if (slices.size() != this->ndim) {
        throw std::invalid_argument("The number of slices must be equal to the number of dimensions.");
    }

    std::vector<int> new_shape;
    std::vector<int> new_stride;
    int new_offset = this->offset_;
    
    for (int i=0; i < this->ndim; i++) {
        int start = slices[i][0], stop = slices[i][1], step = slices[i][2];

        new_shape.push_back((stop - start + (step - 1)) / step);
        new_stride.push_back(step * this->stride_[i]);
        new_offset += start * this->stride_[i];
    }

    // Tensor<dtype> result(std::move(new_shape), std::move(new_stride), std::move(new_offset), this->data_);
    // ERROR!! this->device->getDataPtr() will be double freed !!!
    // Tensor<dtype> result(std::move(new_shape), std::move(new_stride), new_offset, this->device->getDataPtr(), this->device_type);
    // error in this function
    Tensor<dtype> result(std::move(new_shape), std::move(new_stride), new_offset, this->device, this->device_type);
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
    if (value.device_type != this->device_type) {
        throw std::invalid_argument("The device type of value must be equal to the device type of the tensor.");
    }

    // value = value.contiguous();
    if (!is_contiguous(value)) {
        throw std::invalid_argument("The value must be contiguous.");
    }

    // get item first, the new tensor shared the same data with the original tensor in memory.
    slices = process_slices(slices);

    auto out = getItem(slices); // get the item tensor, the out shared the same underlying data with this tensor.

    if (out.shape() != value.shape()) {
        throw std::invalid_argument("The shape of value must be equal to the shape of the slice.");
    }
    
    out.device->setItemEwise(
        value.device->getDataPtr(),
        out.shape_,
        out.stride_,
        out.offset_,
        out.num_elements);
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
    
    out.device->setItemScalar(
        value,
        out.shape_,
        out.stride_,
        out.offset_,
        out.num_elements);
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

    auto prepend_stride = this->stride_;

    if (prepend_shape.size() > new_shape.size()) {
        throw std::invalid_argument("The new shape must be greater than or equal to the original shape.");
    } else if (prepend_shape.size() < new_shape.size()) {
        prepend_shape.insert(prepend_shape.begin(), new_shape.size() - this->shape_.size(), 1);
        prepend_stride.insert(prepend_stride.begin(), new_shape.size() - this->shape_.size(), 0);
    }

    // now prepend_shape.size() equal to new_shape.size()
    std::vector<int> new_stride;
    for (int i=0; i < new_shape.size(); i++) {
        if ((new_shape[i] != prepend_shape[i]) && prepend_shape[i] != 1) {
            throw std::invalid_argument("The dimension to be broadcasted must be 1.");
        }
        new_stride.push_back(prepend_shape[i] == 1 ? 0 : prepend_stride[i]);
    }

    return Tensor<dtype>(std::move(new_shape), std::move(new_stride), this->offset_, this->device, this->device_type);
}

////////////////////////////////////////////////////// unary operations ///////////////////////////////////////////////////////////////////////////////

template <typename dtype>
Tensor<dtype> Tensor<dtype>::applyUnaryOperation(void (Device<dtype>::*func)(dtype*, size_t)) const {
    Tensor<dtype> result(this->shape_, this->device_type);
    (this->device.get()->*func)(result.device->getDataPtr(), result.num_elements);
    return result;
}


template <typename dtype> inline Tensor<dtype> Tensor<dtype>::operator-() const { return applyUnaryOperation(&Device<dtype>::neg); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::sin() const { return applyUnaryOperation(&Device<dtype>::sin); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::cos() const { return applyUnaryOperation(&Device<dtype>::cos); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::exp() const { return applyUnaryOperation(&Device<dtype>::exp); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::log() const { return applyUnaryOperation(&Device<dtype>::log); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::abs() const { return applyUnaryOperation(&Device<dtype>::abs); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::tanh() const { return applyUnaryOperation(&Device<dtype>::tanh); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::silu() const { return applyUnaryOperation(&Device<dtype>::silu); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::sqrt() const { return applyUnaryOperation(&Device<dtype>::sqrt); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::rsqrt() const { return applyUnaryOperation(&Device<dtype>::rsqrt); }


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

    return Tensor<dtype>(std::move(new_shape), std::move(new_stride), this->offset_, this->device, this->device_type);
}

template <typename dtype>
Tensor<dtype> Tensor<dtype>::transpose(int dim0, int dim1) const {
    Tensor<dtype> result = *this;

    std::swap(result.shape_[dim0], result.shape_[dim1]);
    std::swap(result.stride_[dim0], result.stride_[dim1]);

    return result;
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

    Tensor<dtype> result(new_shape, this->device_type);

    int reduce_size = this->shape()[axis];
    
    // Apply the operation for each reduced chunk
    # pragma omp parallel for
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

    Tensor<int> result(new_shape, this->device_type);

    int reduce_size = this->shape()[axis];
    
    // Iterate over the tensor in chunks (based on the reduce size)
    # pragma omp parallel for
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
// template <typename dtype>
// Tensor<dtype> Tensor<dtype>::applyBinaryOperation(const Tensor<dtype>& other, dtype(*op)(dtype, dtype)) const {
//     if (this->device_type != other.device_type) {
//         throw std::invalid_argument("The device type of the two tensors must be the same.");
//     }
// 
//     Tensor<dtype> a = *this, b = other;
//     
//     // implicit broadcasting
//     if (this->shape() != other.shape()) {
//         std::vector<int> shape_a = this->shape();
//         std::vector<int> shape_b = other.shape();
//         auto new_shape = get_broadcast_shape(shape_a, shape_b);
//         a = a.broadcast_to(new_shape); 
//         b = b.broadcast_to(new_shape);
//     }
// 
//     // maybe we can do not call contiguous() to make it faster...
//     /**
//         import torch
//         # Create a base tensor and a non-contiguous tensor
//         a = torch.rand(3, 1)   # Shape (3, 1)
//         b = torch.rand(3, 3).permute(1, 0)  # Shape (3, 3) but non-contiguous
// 
//         # Perform an operation that requires broadcasting
//         result = a + b  # Broadcasting will occur here
// 
//         print(result)
//         print(result.is_contiguous())  # The result might be non-contiguous if any input was non-contiguous
//      */
// 
//     a = a.contiguous(); // if broadcast have influence on the result of contiguous?? (seems no, because contiguous will allocate actually memory for array)
//     b = b.contiguous();
// 
//     Tensor<dtype> result(this->shape(), this->device_type);
//     #pragma omp parallel for
//     for (int i = 0; i < this->num_elements; ++i) {
//         result.data_[i] = op(a.data_[i], b.data_[i]);
//     }
// 
//     return result;
// }
// 
// template <typename dtype> Tensor<dtype> Tensor<dtype>::operator+(const Tensor<dtype>& other) const { return applyBinaryOperation(other, add<dtype>); }
// template <typename dtype> Tensor<dtype> Tensor<dtype>::operator-(const Tensor<dtype>& other) const { return applyBinaryOperation(other, subtract<dtype>); }
// template <typename dtype> Tensor<dtype> Tensor<dtype>::operator*(const Tensor<dtype>& other) const { return applyBinaryOperation(other, multiply<dtype>); }
// template <typename dtype> Tensor<dtype> Tensor<dtype>::operator/(const Tensor<dtype>& other) const { return applyBinaryOperation(other, divide<dtype>); }

/**
 * maybe should support implicit broadcasting 
 * @tparam dtype 
 */
template <typename dtype>
Tensor<dtype> Tensor<dtype>::applyBinaryOperation(const Tensor<dtype>& other, dtype(*op)(dtype, dtype)) const {
    if (this->device_type != other.device_type) {
        throw std::invalid_argument("The device type of the two tensors must be the same.");
    }

    Tensor<dtype> a = *this, b = other;
    
    // implicit broadcasting
    if (this->shape() != other.shape()) {
        std::vector<int> shape_a = this->shape();
        std::vector<int> shape_b = other.shape();
        auto new_shape = get_broadcast_shape(shape_a, shape_b);
        a = a.broadcast_to(new_shape); 
        b = b.broadcast_to(new_shape);
    }

    // maybe we can do not call contiguous() to make it faster...
    /**
        import torch
        # Create a base tensor and a non-contiguous tensor
        a = torch.rand(3, 1)   # Shape (3, 1)
        b = torch.rand(3, 3).permute(1, 0)  # Shape (3, 3) but non-contiguous

        # Perform an operation that requires broadcasting
        result = a + b  # Broadcasting will occur here

        print(result)
        print(result.is_contiguous())  # The result might be non-contiguous if any input was non-contiguous
     */

    a = a.contiguous();
    b = b.contiguous();

    Tensor<dtype> result(this->shape(), this->device_type);
    #pragma omp parallel for
    for (int i = 0; i < this->num_elements; ++i) {
        result.data_[i] = op(a.data_[i], b.data_[i]);
    }

    return result;
}

template <typename dtype> Tensor<dtype> Tensor<dtype>::operator+(const Tensor<dtype>& other) const { return applyBinaryOperation(other, add<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator-(const Tensor<dtype>& other) const { return applyBinaryOperation(other, subtract<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator*(const Tensor<dtype>& other) const { return applyBinaryOperation(other, multiply<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator/(const Tensor<dtype>& other) const { return applyBinaryOperation(other, divide<dtype>); }


/**
 * if this is a broadcasted tensor, need to use contiguous() firtst, or the num_elements is not the actual elem size of the tensor data_.
 * @tparam dtype 
 */
template <typename dtype>
Tensor<dtype> Tensor<dtype>::applyBinaryScalarOperation(dtype scalar, dtype(*op)(dtype, dtype)) const {
    // Tensor<dtype> result = this->contiguous();
    Tensor<dtype> result(this->shape(), this->device_type);
    Tensor<dtype> this_contiguous = this->contiguous();

    #pragma omp parallel for
    for (int i = 0; i < result.num_elements; ++i) { // broadcast may get error, for it does not have num_elements elems actually.
        // result.data_[i] = op(result.data_[i], scalar);
        result.data_[i] = op(this_contiguous.data_[i], scalar);
    }

    return result;
}

template <typename dtype> Tensor<dtype> Tensor<dtype>::operator+(dtype scalar) const { return applyBinaryScalarOperation(scalar, add<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator-(dtype scalar) const { return applyBinaryScalarOperation(scalar, subtract<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator*(dtype scalar) const { return applyBinaryScalarOperation(scalar, multiply<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator/(dtype scalar) const { return applyBinaryScalarOperation(scalar, divide<dtype>); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::pow(dtype scalar) const { return applyBinaryScalarOperation(scalar, power<dtype>); }

/** can be called in gdb, operator << can not be called... */
void print_tensor_float(const Tensor<float>& tensor) {
    std::cout << tensor << std::endl << std::endl;
}
