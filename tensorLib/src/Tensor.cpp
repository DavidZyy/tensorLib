#include "Tensor.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <iomanip>
#include "iostream"
#include "math.h"
#include "omp.h"
#include "device/cpu/CPU.hpp"
#include "device/cuda/CUDA.cuh"
#include "device/Device.hpp"
#include "ops/matmul.hpp"

// Explicit instantiation
template class Tensor<int>;
template class Tensor<float>;
template class Tensor<half>;
template class Tensor<int8_t>;

template <typename dtype>
Tensor<dtype>::Tensor(const std::vector<int>& shape, const std::string& device_type) : ndim(shape.size()), shape_(shape), offset_(0), device_type(device_type) {
        num_elements = 1; // even shape is empty, it should have 1 elem, means a scala.
        for (int dim : shape) {
            num_elements *= dim;
        }

        // this->data_ = std::shared_ptr<dtype[]>(new dtype[num_elements], Deleter<dtype>(num_elements));

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

/************************************************************************************************************************************************************/

template <typename dtype>
void Tensor<dtype>::printTensor(std::ostream& os, size_t depth, std::vector<int> indices) const {
    if (depth == ndim - 1) {
        os << "[";
        auto idx = 0;
        for (auto& dim: indices)
            idx += dim;

        for (int i = 0; i < shape_[depth]; ++i) {
            if constexpr (std::is_same_v<dtype, half>) {
                if (i >= 0 && i <= 2 ) {
                    os << std::setw(3) << __half2float(this->device->getDataLinear(idx + i*stride_[depth] + offset_)) << ", ";
                } else if ( i>= shape_[depth]-3 && i <= shape_[depth]-1) {
                    os << std::setw(3) << __half2float(this->device->getDataLinear(idx + i*stride_[depth] + offset_)) << ", ";
                } else if (i == 3){
                    os << "..., ";
                } else {
                    // not print anything
                }
            } else {

                // if (i > 0) os << ", ";
                // os << std::setw(3) << this->device->getDataLinear(idx + i*stride_[depth] + offset_);

                // for debug output
                if (i >= 0 && i <= 2 ) {
                    os << std::setw(3) << this->device->getDataLinear(idx + i*stride_[depth] + offset_) << ", ";
                } else if ( i>= shape_[depth]-3 && i <= shape_[depth]-1) {
                    os << std::setw(3) << this->device->getDataLinear(idx + i*stride_[depth] + offset_) << ", ";
                } else if (i == 3){
                    os << "..., ";
                } else {
                    // not print anything
                }
            }
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
Tensor<dtype> Tensor<dtype>::matmul(const Tensor<dtype>& other, std::optional<Tensor<dtype>> result_opt) const {
    // return ops::matmul<dtype>::call(*this, other);
    return ops::matmul<dtype>::call2(*this, other, result_opt);
}

/**
 * view use the same data as the original tensor, and reshape copy the data.
 * @tparam dtype 
 */
template <typename dtype>
Tensor<dtype> Tensor<dtype>::view(const std::vector<int>& new_shape) const {
    if (new_shape == this->shape()) 
        return *this;

    if (!is_contiguous(*this))
        throw std::invalid_argument("This tensor is not contiguous.");

    int num = 1;
    for (auto i=0; i < new_shape.size(); i++) {
        num *= new_shape[i];
    }
    if (num != this->num_elements) {
        throw std::invalid_argument("The number of elements is not equal.");
    }

    int new_dim = new_shape.size(); 
    std::vector<int> new_stride = std::vector<int>(new_dim);

    // *this is contiguous, so we can calculate the new stride like this
    if (new_dim > 0) {
        new_stride[new_dim - 1] = 1;
        for (int i = new_dim - 2; i >= 0; --i) {
            new_stride[i] = new_stride[i + 1] * new_shape[i + 1];
        }
    }
    Tensor<dtype> result(std::move(new_shape), std::move(new_stride), this->offset_, this->device, this->device_type);

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

// add a flag to choose whether make it row contiguous or column contiguous,
// default is row contiguous.
template <typename dtype>
Tensor<dtype> Tensor<dtype>::contiguous(std::optional<Tensor<dtype>> result_opt) const {
    if (is_contiguous(*this)) {
        return *this;
    }

    // Tensor<dtype> result(this->shape(), this->device_type);
    Tensor<dtype> result;
    if (result_opt.has_value()) {
        result = result_opt.value();
        if (result.shape() != this->shape()) {
            throw std::invalid_argument("result shape does not match new_shape");
        }
    } else {
        result = Tensor<dtype> (this->shape(), this->device_type);
    }

    this->device->contiguous(
        result.device->getDataPtr(), 
        this->shape_,
        this->stride_,
        this->offset_,
        this->num_elements);

    return result;
}


template <typename dtype>
Tensor<dtype> Tensor<dtype>::getItem(std::vector<std::vector<int>>& slices) const {
    // assert(this->shape().size() == this->ndim);
    auto ps = process_slices(slices); // processed slices(ps)

    if (ps.size() != this->ndim) {
        throw std::invalid_argument("The number of slices must be equal to the number of dimensions.");
    }

    std::vector<int> new_shape;
    std::vector<int> new_stride;
    int new_offset = this->offset_;
    
    for (int i=0; i < this->ndim; i++) {
        int start = ps[i][0], stop = ps[i][1], step = ps[i][2];

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
// void Tensor<dtype>::setItem(std::vector<std::vector<int>>& slices, const Tensor<dtype>& value) {
void Tensor<dtype>::setItem(std::vector<std::vector<int>>& slices, const Tensor<dtype> value) {
// void Tensor<dtype>::setItem(std::vector<std::vector<int>>& slices, const Tensor<dtype> value) {
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
        value.device->getDataPtr() + value.offset_,
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
template <void (Device<dtype>::*func)(dtype*, size_t)>
Tensor<dtype> Tensor<dtype>::applyUnaryOperation(std::optional<Tensor<dtype>> result_opt) const {
    // Tensor<dtype> result(this->shape_, this->device_type);
    Tensor<dtype> result;
    if (result_opt.has_value()) {
        result = result_opt.value();
        if (result.shape() != this->shape()) {
            throw std::invalid_argument("The shape of the result must be the same as the shape of the tensor.");
        }
    } else {
        result = Tensor<dtype>(this->shape(), this->device_type);
    }

    (this->device.get()->*func)(result.device->getDataPtr(), result.num_elements);
    return result;
}

template <typename dtype> inline Tensor<dtype> Tensor<dtype>::operator-() const { return applyUnaryOperation<&Device<dtype>::neg>(); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::sin() const { return applyUnaryOperation<&Device<dtype>::sin>(); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::cos() const { return applyUnaryOperation<&Device<dtype>::cos>(); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::exp() const { return applyUnaryOperation<&Device<dtype>::exp>(); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::log() const { return applyUnaryOperation<&Device<dtype>::log>(); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::abs() const { return applyUnaryOperation<&Device<dtype>::abs>(); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::tanh() const { return applyUnaryOperation<&Device<dtype>::tanh>(); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::silu(std::optional<Tensor<dtype>> result_opt) const { return applyUnaryOperation<&Device<dtype>::silu>(result_opt); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::sqrt() const { return applyUnaryOperation<&Device<dtype>::sqrt>(); }
template <typename dtype> inline Tensor<dtype> Tensor<dtype>::rsqrt() const { return applyUnaryOperation<&Device<dtype>::rsqrt>(); }



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

////////////////////////////////////////////////////// softmax operations ///////////////////////////////////////////////////////////////////////////////
// support not last dimension softmax
// template<typename dtype>
// Tensor<dtype> Tensor<dtype>::softmax(int dim, std::optional<Tensor<dtype>> result_opt) const {
//     std::vector<int> axis0, axis1;
//     for (int i=0; i < this->ndim; i++)
//         if (i != dim) axis0.push_back(i);
//     axis0.push_back(dim);
// 
//     for (int i=0; i < this->ndim; i++) {
//         if (i < dim) axis1.push_back(i);
//         else if (i == dim) axis1.push_back(this->ndim - 1);
//         else axis1.push_back(i - 1);
//     }
// 
//     Tensor<dtype> a = this->permute(axis0);
//     a = a.contiguous();
// 
//     Tensor<dtype> result(a.shape_, a.device_type);
//     int cols = a.shape_[a.ndim-1];
//     int rows = a.num_elements / cols;
//     a.device->softmax(result.device->getDataPtr(), rows, cols);
// 
//     Tensor<dtype> b = result.permute(axis1);
//     b = b.contiguous();
// 
//     return b;
// }

template<typename dtype>
Tensor<dtype> Tensor<dtype>::softmax(int dim, std::optional<Tensor<dtype>> result_opt) const {
    if (!is_contiguous(*this)) {
        throw std::invalid_argument("This tensor is not contiguous.");
    }
    if (dim != this->ndim - 1) {
        throw std::invalid_argument("The dimension to be softmaxed must be the last dimension.");
    }

    // Tensor<dtype> result(this->shape(), this->device_type);
    Tensor<dtype> result;
    if (result_opt.has_value()) {
        result = result_opt.value();
        if (result.shape() != this->shape()) {
            throw std::invalid_argument("The shape of the result must be the same as the shape of the tensor.");
        }
    } else {
        result = Tensor<dtype>(this->shape(), this->device_type);
    }

    int cols = this->shape_[this->ndim-1];
    int rows = this->num_elements / cols;
    this->device->softmax(result.device->getDataPtr(), rows, cols);
    return result;
}
////////////////////////////////////////////////////// binary operations ///////////////////////////////////////////////////////////////////////////////
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
template <void (Device<dtype>::*func)(dtype*, dtype*, size_t) const>
Tensor<dtype> Tensor<dtype>::applyBinaryOperation(const Tensor<dtype>& other, std::optional<Tensor<dtype>> result_opt) const {
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

    // Tensor<dtype> result(this->shape(), this->device_type);
    Tensor<dtype> result;
    if (result_opt.has_value()) {
        result = result_opt.value();
        if (result.shape() != this->shape()) {
            throw std::invalid_argument("The shape of the result must be the same as the shape of the tensor.");
        }
    } else {
        result = Tensor<dtype>(this->shape(), this->device_type);
    }


    // (this->device.get()->*func)(result.device->getDataPtr(), other.device->getDataPtr(), result.num_elements); // error!!
    (a.device.get()->*func)(result.device->getDataPtr(), b.device->getDataPtr(), result.num_elements);
    return result;
}

template <typename dtype> Tensor<dtype> Tensor<dtype>::operator+(const Tensor<dtype>& other) const { return applyBinaryOperation<&Device<dtype>::add>(other); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator-(const Tensor<dtype>& other) const { return applyBinaryOperation<&Device<dtype>::sub>(other); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator*(const Tensor<dtype>& other) const { return applyBinaryOperation<&Device<dtype>::mul>(other); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator/(const Tensor<dtype>& other) const { return applyBinaryOperation<&Device<dtype>::div>(other); }

template <typename dtype> Tensor<dtype> Tensor<dtype>::EwiseAdd(const Tensor<dtype>& other, std::optional<Tensor<dtype>> result_opt) const { return applyBinaryOperation<&Device<dtype>::add>(other, result_opt); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::EwiseMul(const Tensor<dtype>& other, std::optional<Tensor<dtype>> result_opt) const { return applyBinaryOperation<&Device<dtype>::mul>(other, result_opt); }
/**
 * if this is a broadcasted tensor, need to use contiguous() firtst, or the num_elements is not the actual elem size of the tensor' device data_.
 * @tparam dtype 
 */
template <typename dtype>
template <void (Device<dtype>::*func)(dtype*, dtype, size_t) const>
Tensor<dtype> Tensor<dtype>::applyBinaryScalarOperation(dtype scalar, std::optional<Tensor<dtype>> result_opt) const {

    // Tensor<dtype> result(this->shape(), this->device_type);
    Tensor<dtype> result;
    if (result_opt.has_value()) {
        result = result_opt.value();
        if (result.shape() != this->shape()) {
            throw std::invalid_argument("The shape of the result must be the same as the shape of the tensor.");
        }
    } else {
        result = Tensor<dtype>(this->shape(), this->device_type);
    }

    Tensor<dtype> this_contiguous = this->contiguous();//if do not contiguous broadcast may get error, for it does not have num_elements elems actually.
    (this_contiguous.device.get()->*func)(result.device->getDataPtr(), scalar, result.num_elements);
    return result;
}

template <typename dtype> Tensor<dtype> Tensor<dtype>::operator+(dtype scalar) const { return applyBinaryScalarOperation<&Device<dtype>::add>(scalar); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator-(dtype scalar) const { return applyBinaryScalarOperation<&Device<dtype>::sub>(scalar); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator*(dtype scalar) const { return applyBinaryScalarOperation<&Device<dtype>::mul>(scalar); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::operator/(dtype scalar) const { return applyBinaryScalarOperation<&Device<dtype>::div>(scalar); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::pow(dtype scalar) const { return applyBinaryScalarOperation<&Device<dtype>::pow>(scalar); }

template <typename dtype> Tensor<dtype> Tensor<dtype>::ScalarDiv(dtype scalar, std::optional<Tensor<dtype>> result_opt) const { return applyBinaryScalarOperation<&Device<dtype>::div>(scalar, result_opt); }
template <typename dtype> Tensor<dtype> Tensor<dtype>::ScalarAdd(dtype scalar, std::optional<Tensor<dtype>> result_opt) const { return applyBinaryScalarOperation<&Device<dtype>::add>(scalar, result_opt); }

/******************************************************************************************************************************/

/** can be called in gdb, operator << can not be called... */
void print_tensor_float(const Tensor<float>& tensor) {
    std::cout << tensor << std::endl << std::endl;
}

void print_tensor_half(const Tensor<half>& tensor) {
    std::cout << tensor << std::endl << std::endl;
}

void print_tensor_int(const Tensor<int>& tensor) {
    std::cout << tensor << std::endl << std::endl;
}

// can not be called in gdb
// template <typename dtype>
// void print_tensor(const Tensor<dtype>& tensor) {
//     std::cout << tensor << std::endl << std::endl;
// }
// 
// // Explicit instantiation for common types (optional, but can help with reducing compilation time)
// template void print_tensor<float>(const Tensor<float>& tensor);
// template void print_tensor<half>(const Tensor<half>& tensor);
// template void print_tensor<int>(const Tensor<int>& tensor);
