#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>
#include <ostream>
#include <atomic>
#include "CPU.hpp"
#include "CUDA.hpp"
#include "Device.hpp"

static std::atomic<size_t> memoryUsage(0);

template <typename dtype>
class Deleter {
public:
    void operator()(dtype* ptr) const {
        delete[] ptr;
        auto a = memoryUsage.load(); // in one case, a = 0, another a is not 0,
        auto b = sizeof(dtype) * elem_num;
        memoryUsage = a - b;
        // memoryUsage -= sizeof(dtype) * elem_num;
        // std::cout << "Free: " << sizeof(dtype) * elem_num << ", now: " << memoryUsage << std::endl;
    }

    Deleter(size_t elem_num) : elem_num(elem_num) {}

private:
    size_t elem_num;
};

template <typename dtype>
class Tensor {
public:
    // Constructor
    Tensor() = default;
    Tensor(const std::vector<int>& shape, const std::string& device_type = "cpu");
    Tensor(const std::vector<int>& shape, const std::shared_ptr<dtype[]>& data, const std::string& device_type = "cpu");
    Tensor(const std::vector<int>&& shape, const std::vector<int> &&stride, const int &offset, const std::shared_ptr<dtype[]>& data, const std::string& device_type = "cpu");
    // Tensor(const std::vector<int>&& shape, const std::vector<int> &&stride, const int &offset, dtype *data_ptr, const std::string& device_type = "cpu");
    Tensor(const std::vector<int>& shape, const std::shared_ptr<Device<dtype>>& device, const std::string& device_type = "cpu");
    Tensor(const std::vector<int>&& shape, const std::vector<int> &&stride, const int &offset, const std::shared_ptr<Device<dtype>>& device, const std::string& device_type = "cpu");
    template<typename OtherType> Tensor(const Tensor<OtherType>& other); // support static cast

    // Destructor
    ~Tensor();

    // print methed // Declaration of friend function
    template <typename T>
    friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor);
    template <typename T>
    friend std::ostream& outputTensor(std::ostream& os, const Tensor<T>& tensor);

    // Method to get shape
    const std::vector<int>& shape() const {
        return shape_;
    }

    // Method to get data (double is used as an example type)
    // const std::shared_ptr<dtype[]> data() const {
    //     return data_;
    // }

    // inline const dtype& getData(const std::vector<int>& indices) const;
    inline const dtype getData(const std::vector<int>& indices) const;
    inline void setData(const std::vector<int>& indices, const dtype& value);

    // Accessor and modifier for tensor elements (non-const version)
    dtype& operator()(const std::vector<int>& indices);
    // Accessor for tensor elements (const version)
    const dtype& operator()(const std::vector<int>& indices) const;

    Tensor<int> operator==(const Tensor<dtype>& other) const;


    // Overloaded operator[] to return TensorProxy for nested indexing
    // need to return a new thensor with different shape_, stride_, offset_, ndim, but have the same data_ area.
    // TensorProxy<dtype> operator[](int idx) {
    //     return TensorProxy(*this, {idx});
    // }

    // Matrix multiplication method
    Tensor<dtype> matmul(const Tensor<dtype>& other) const;

    Tensor<dtype> view(const std::vector<int>& shape) const;

    // startIdx <= idx < endIdx
    Tensor<dtype> slice(int startIdx, int endIdx, int dim) const;

    /* Slices the input tensor along the selected dimension at the given index. 
        This function returns a view of the original tensor with the given dimension removed.*/
    Tensor<dtype> select(int dim, int index) const;

    template<typename T>
    friend Tensor<T> maximum(Tensor<T> a, Tensor<T> b);

    template<typename T> friend Tensor<T> full(const std::vector<int>& shape, T fill_value, const std::string& device);
    template<typename T> friend Tensor<T> zeros(const std::vector<int>& shape);
    template<typename T> friend Tensor<T> ones(const std::vector<int>& shape);
    template<typename T> friend Tensor<T> randn(const std::vector<int>& shape, T mean, T std);

    template<typename T>
    friend Tensor<T> apply_rotary_emb(Tensor<T> &input, Tensor<T> &freqs, int start_pos);

    Tensor<dtype> transpose(int dim0, int dim1) const;
    Tensor<dtype> permute(const std::vector<int>& new_axes) const;

    // return a new tensor with the same shape and data, 
    // but with a different memory layout which is contiguous.
    Tensor<dtype> contiguous() const;

    // get or set a sub-tensor of this tensor. The implementation here refers to the homework project of CMU10_414.
    Tensor<dtype> getItem(std::vector<std::vector<int>>& slices) const;
    void setItem(std::vector<std::vector<int>>& slices, const Tensor<dtype> value);
    // void setItem(std::vector<std::vector<int>>& slices, const Tensor<dtype>& value);
    void setItem(std::vector<std::vector<int>>& slices, dtype value);

    Tensor<dtype> broadcast_to(const std::vector<int>& new_shape) const;

    // unary methods
    Tensor<dtype> operator-() const;  // negative operator
    Tensor<dtype> sin() const;
    Tensor<dtype> cos() const;
    Tensor<dtype> exp() const;
    Tensor<dtype> log() const;
    Tensor<dtype> abs() const;
    Tensor<dtype> tanh() const;
    Tensor<dtype> silu() const;
    Tensor<dtype> sqrt() const;
    Tensor<dtype> rsqrt() const;

    // binary methods
    Tensor<dtype> operator+(const Tensor<dtype>& other) const;
    Tensor<dtype> operator-(const Tensor<dtype>& other) const;
    Tensor<dtype> operator*(const Tensor<dtype>& other) const;
    Tensor<dtype> operator/(const Tensor<dtype>& other) const;
    Tensor<dtype> operator+(dtype scalar) const; // could support Tensor + 1(not a lvalue), (dtype& scalar) can not support this
    Tensor<dtype> operator-(dtype scalar) const;
    Tensor<dtype> operator*(dtype scalar) const;
    Tensor<dtype> operator/(dtype scalar) const;
    Tensor<dtype> pow(dtype scalar) const;

    // reduce methods(reduce 1 dimension each function call), like sum, max
    Tensor<dtype> max (std::optional<int> axis = {}, bool keepdims = false) const;
    Tensor<dtype> min (std::optional<int> axis = {}, bool keepdims = false) const;
    Tensor<dtype> sum (std::optional<int> axis = {}, bool keepdims = false) const;
    Tensor<dtype> mean(std::optional<int> axis = {}, bool keepdims = false) const;
    Tensor<int> argmax(std::optional<int> axis = {}, bool keepdim = false) const;
    Tensor<int> argmin(std::optional<int> axis = {}, bool keepdim = false) const;

    Tensor<dtype> softmax(int dim) const;

    // Tensor<dtype> apply_rotary_emb(Tensor<dtype> &input, Tensor<dtype> &freqs, int start_pos);

    /* data is managed by copy on write (COW) later */
    // std::vector<dtype> data_;
    // std::shared_ptr<dtype[]> data_;
    // int data_size; // may different from num_elements in broadcasted Tensor, used for memory safety
    int num_elements;

    // used for quantize
    // int group_size;  // seem as one group now for simple
    float scale;

    // the offset of data_, used for slice method to share the same memory area of data_.
    int offset_;
    std::vector<int> stride_;
    int ndim;
    std::vector<int> shape_;
    std::string device_type;
    std::shared_ptr<Device<dtype>> device;

    /**
     * seems this shape msethod can handle non-contiguous Tensor, both this and below can be used in matmul(contiguous?),
     *  but below stride method seems can not used in setItem or contiguous method(no contiguous?).
     */
    inline std::vector<int> getIndicesFromLinearIndex(size_t linear_index) const {
        // assert(this->shape_.size() == this->ndim);
        // std::vector<int> indices(this->ndim);
        std::vector<int> indices(this->shape_.size());
        
        // Iterate from the last dimension to the first(0 dim), because the data is stored contiguously form last dim(the last dim's stride is 1).
        // for (int i = this->ndim - 1; i >= 0; --i) {
        for (int i = this->shape_.size()-1; i >= 0; --i) {
            indices[i] = linear_index % shape_[i];
            linear_index /= shape_[i];
        }
    
        return indices;
    }
// private:

    // helper method for operator<<
    void printTensor(std::ostream& os, size_t depth, std::vector<int> indices) const;
    // Helper function to calculate linear index from multi-dimensional indices
    inline size_t calculateLinearIndex(const std::vector<int>& indices) const;
    // helper function for view
    bool is_contiguous(const Tensor<dtype>& t) const;
    // used in getItem method, process slice, supplement abbreviation of slice to full
    std::vector<std::vector<int>> process_slices(const std::vector<std::vector<int>>& slices) const;

    // reduce methods helper
    int handle_axis(int axis) const;
    Tensor<dtype> get_reduce_view(int axis) const;
    std::vector<int> get_reduce_shape(int axis, bool keepdims) const;

    std::vector<int> get_broadcast_shape(std::vector<int>& shape_a, std::vector<int>& shape_b) const; // without const, will cause error.

    template <void (Device<dtype>::*func)(dtype*, size_t)>
    Tensor<dtype> applyUnaryOperation() const;
    template <void (Device<dtype>::*func)(dtype*, dtype*, size_t) const >
    Tensor<dtype> applyBinaryOperation(const Tensor<dtype>& other) const;
    template <void (Device<dtype>::*func)(dtype*, dtype, size_t) const >
    Tensor<dtype> applyBinaryScalarOperation(dtype scalar) const;
    template <void (Device<dtype>::*func)(dtype*, size_t, size_t) const>
    Tensor<dtype> reduceOperation(std::optional<int> axis, bool keepdims) const;
    template <void (Device<dtype>::*func)(int*, size_t, size_t) const>
    Tensor<int> reduceOperationArg(std::optional<int> axis, bool keepdims) const;

    /**
     * fuse getIndicesFromLinearIndex and calculateLinearIndex
     */
    inline size_t convertIdx(size_t linear_index) const {
        size_t linear_index_new = 0;

        for (int i = this->ndim - 1; i >= 0; --i) {
            int cur_dim_id = linear_index % this->shape_[i];
            linear_index /= this->shape_[i];
            linear_index_new += cur_dim_id * this->stride_[i];
        }

        return linear_index_new + this->offset_;
    }

    // Device<dtype> device; // error, abstract class can not be member

    // num_elements should not put in device, maybe should put in Tensor ...
};

// Definition of the conversion constructor outside the class
template <typename dtype>  // This is the Tensor's dtype template
template <typename OtherType>  // This is the type we are converting from
Tensor<dtype>::Tensor(const Tensor<OtherType>& other) {
    this->num_elements = other.num_elements;  // Copy the number of elements
    this->offset_ = other.offset_;
    this->stride_ = other.stride_;
    this->ndim = other.ndim;
    this->shape_ = other.shape();  // Copy the shape
    this->device_type = other.device_type;
    
    if (device_type == "cpu") {
        this->device = std::shared_ptr<CPU<dtype>>(new CPU<dtype>(num_elements));
    } else if (device_type == "cuda") {
        this->device = std::shared_ptr<CUDA<dtype>>(new CUDA<dtype>(num_elements));
    } else {
        throw std::invalid_argument("Invalid device name");
    }

    Tensor<OtherType> other_contiguous = other.contiguous();

    for (int i = 0; i < other_contiguous.num_elements; ++i) {
        dtype a = static_cast<dtype>(other_contiguous.device->getDataLinear(i));
        this->device->setDataLinear(i, a);
    }
}

// Overload operator<< to print Tensor
template <typename dtype>
std::ostream& operator<<(std::ostream& os, const Tensor<dtype>& tensor) {
    const auto& shape = tensor.shape();
    // const auto& data = tensor.data();

    // scalar
    if (shape.size() == 0) {
        // os << "[]";
        // os << tensor.data_[0];
        os << tensor.device->getDataLinear(0);
    } else {
        tensor.printTensor(os, 0, {});
    }

    return os;
}

// /**
//  * same as operator<<, used for debug, since operator<< can not be called in vscode DEBUG CONSOLE.
//  * @tparam dtype 
//  */
// template <typename dtype>
// std::ostream& outputTensor(std::ostream& os, const Tensor<dtype>& tensor) {
//     const auto& shape = tensor.shape();
//     const auto& data = tensor.data();
// 
//     if (shape.size() == 0) {
//         // os << "[]";
//         os << tensor.data_[0];
//     } else {
//         tensor.printTensor(os, 0, {});
//     }
// 
//     return os;
// }

template <typename T>
Tensor<T> maximum(Tensor<T> a, Tensor<T> b) {
    // assume b is a scala.
    assert(b.shape().empty());

    Tensor<T> result(a.shape());
    for (auto i = 0; i < a.num_elements; ++i) {
        result.data_[i] = std::max(a.data_[i], b.data_[0]);
    }

    return result;
}

template <typename dtype>
Tensor<dtype> zeros(const std::vector<int>& shape) {
    Tensor<dtype> result = Tensor<dtype>(shape);
    for(auto i = 0; i < result.num_elements; ++i) {
        result.data_[i] = 0;
    }
    return result;
}

// template <typename dtype>
// Tensor<dtype> ones(const std::vector<int>& shape) {
//     Tensor<dtype> result = Tensor<dtype>(shape);
//     for(auto i = 0; i < result.num_elements; ++i) {
//         result.data_[i] = 1;
//     }
//     return result;
// }

template <typename dtype>
Tensor<dtype> full(const std::vector<int>& shape, dtype fill_value, const std::string& device = "cpu") {
    Tensor<dtype> result = Tensor<dtype>(shape, device);
    result.device->full(result.num_elements, fill_value);
    return result;
}

/**
 * return a tensor filled with random numbers from a normal distribution, with mean 0 and variance 1.
 * @tparam dtype 
 */
template <typename dtype>
Tensor<dtype> randn(const std::vector<int>& shape, const std::string& device = "cpu") {
    Tensor<dtype> result = Tensor<dtype>(shape, device);
    result.device->randn(result.num_elements); 
    return result;
}

template <typename dtype>
bool Tensor<dtype>::is_contiguous(const Tensor<dtype>& t) const {
    int stride = 1;
    for(int i = t.ndim - 1; i >= 0; --i) {
        if(stride != t.stride_[i])
            return false;
        stride *= t.shape_[i];
    }
    return true;
}

/**
 * freqs is a 2d tensor with shape [T, head_dim/2]
 * freqs = m[theta_1, theta_2, ..., theta_(d/2)], m = 1, 2, ..., T
 * theta_i = 10000 ^ {-2(i-1)/d}
 * 
 * input's shape is [B, T, n_heads, head_dim]
 */
template <typename dtype>
Tensor<dtype> apply_rotary_emb(Tensor<dtype> &input, int start_pos) {
    if (input.shape().size() != 4) {
        throw std::invalid_argument("Invalid shape.");
    }

    int B = input.shape()[0];
    int T = input.shape()[1];
    int n_heads = input.shape()[2];
    int head_dim = input.shape()[3];
    
    input = input.contiguous();

    Tensor<dtype> result(input.shape(), input.device_type);

    int H = B*T*n_heads;
    int W = head_dim;
    input.device->apply_rotary_emb(
        input.device->getDataPtr(),
        result.device->getDataPtr(),
        start_pos,
        H,
        W
    );

    return result;
}

template <typename dtype>
inline size_t Tensor<dtype>::calculateLinearIndex(const std::vector<int>& indices) const{
    // doulble check
    // if (indices.size() != shape_.size() || indices.size() != ndim) {
    //     throw std::invalid_argument("Error: Indices size does not match tensor dimension");
    // }
    // assert(indices.size() == this->ndim);

    size_t linear_index = 0;
    // for (size_t i = 0; i < this->ndim; ++i) {
    for (size_t i = 0; i < indices.size(); ++i) {
        // if (indices[i] < 0 || indices[i] >= shape_[i]) {
        //     throw std::out_of_range("Error: Index out of range");
        // }
        linear_index += indices[i] * this->stride_[i];
    }

    return linear_index + this->offset_;
}

/**
 * maybe should return a Tensor wrapped the data, which is done by pytorch.
 * @tparam dtype 
 */
template <typename dtype>
// get segment fault return dtype&
// inline const dtype& Tensor<dtype>::getData(const std::vector<int>& indices) const {
inline const dtype Tensor<dtype>::getData(const std::vector<int>& indices) const {
    size_t linear_index = calculateLinearIndex(indices);

    // return data_[linear_index];
    return this->device->getDataLinear(linear_index);
}

// Implementation of setData method
template <typename dtype>
inline void Tensor<dtype>::setData(const std::vector<int>& indices, const dtype& value) {
    size_t linear_index = calculateLinearIndex(indices);

    // data_[linear_index] = value;
    this->device->setDataLinear(linear_index, value);
}
