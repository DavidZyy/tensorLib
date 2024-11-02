#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <ostream>
#include <atomic>
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
    Tensor(const std::vector<int>& shape, const std::string& device = "cpu");
    Tensor(const std::vector<int>& shape, const std::shared_ptr<dtype[]>& data, const std::string& device = "cpu");
    Tensor(const std::vector<int>&& shape, const std::vector<int> &&stride, const int &offset, const std::shared_ptr<dtype[]>& data, const std::string& device = "cpu");
    Tensor(const std::vector<int>&& shape, const std::vector<int> &&stride, const int &offset, dtype *data_ptr, const std::string& device = "cpu");
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
    const std::shared_ptr<dtype[]> data() const {
        return data_;
    }

    inline const dtype& getData(const std::vector<int>& indices) const;
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
    friend Tensor<T> apply_rotary_emb(Tensor<T> input, Tensor<T> freqs, int start_pos);

    Tensor<dtype> transpose(int dim0, int dim1) const;
    Tensor<dtype> permute(const std::vector<int>& new_axes) const;

    // return a new tensor with the same shape and data, 
    // but with a different memory layout which is contiguous.
    Tensor<dtype> contiguous() const;

    // get or set a sub-tensor of this tensor. The implementation here refers to the homework project of CMU10_414.
    Tensor<dtype> getItem(std::vector<std::vector<int>>& slices) const;
    void setItem(std::vector<std::vector<int>>& slices, const Tensor<dtype>& value);
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
    Tensor<dtype> max(int axis, bool keepdims = false) const;
    Tensor<dtype> min(int axis, bool keepdims = false) const;
    Tensor<dtype> sum(int axis, bool keepdims = false) const;
    Tensor<dtype> mean(int axis, bool keepdims = false) const;
    Tensor<int> argmax(int dim, bool keepdim = false) const;
    Tensor<int> argmin(int dim, bool keepdim = false) const;

    Tensor<dtype> softmax(int dim) const;

    // int8_t quantize, but use int32_t store value now in case of overflow when perform mutmul.
    Tensor<int> quantize() const;
    Tensor<float> dequantize() const;

    /* data is managed by copy on write (COW) later */
    // std::vector<dtype> data_;
    std::shared_ptr<dtype[]> data_;
    int data_size; // may different from num_elements in broadcasted Tensor, used for memory safety
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
private:

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

    // helper function for ewise or scalar methods
    Tensor<dtype> applyBinaryOperation(const Tensor<dtype>& other, dtype(*op)(dtype, dtype)) const;
    Tensor<dtype> applyBinaryScalarOperation(dtype scalar, dtype(*op)(dtype, dtype)) const;

    std::vector<int> get_broadcast_shape(std::vector<int>& shape_a, std::vector<int>& shape_b) const; // without const, will cause error.

    // helper function for reduce methods
    Tensor<dtype> reduce(int axis, bool keepdims, dtype(*op)(dtype, dtype)) const;
    Tensor<int> reduce_arg(int axis, bool keepdims, bool(*comp)(dtype, dtype)) const;

    // General template for inline unary operations
    inline Tensor<dtype> applyUnaryOperation(dtype (*func)(dtype)) const {
        Tensor<dtype> result(this->shape_);
        
        // Inline parallel loop for performance (OpenMP enabled)
        #pragma omp parallel for
        for (size_t i = 0; i < this->num_elements; ++i) {
            result.data_[i] = func(this->data_[i]);  // Apply function to each element
        }
        
        return result;
    }

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
    Tensor<dtype> tmp(other.shape());
    // Convert the data from 'OtherType' to 'dtype'
    this->num_elements = other.num_elements;  // Copy the number of elements
    this->offset_ = other.offset_;
    this->stride_ = other.stride_;
    this->ndim = other.ndim;
    this->shape_ = other.shape();  // Copy the shape
    // data_ = std::make_shared<dtype[]>(num_elements);  // Allocate memory for new data
    // data_ = std::shared_ptr<dtype[]>(new dtype[this->num_elements], Deleter<dtype>(num_elements));
    // memoryUsage += num_elements * sizeof(dtype);
    // std::cout << "Allocate: " << sizeof(dtype) * num_elements << ", now: " << memoryUsage << std::endl;
    this->data_ = tmp.data_;

    // Element-wise conversion from OtherType to dtype
    for (int i = 0; i < num_elements; ++i) {
        data_[i] = static_cast<dtype>(other.data_[i]);
    }
}

template <typename dtype>
static inline dtype add(dtype a, dtype b) {
    return a + b;
}

template <typename dtype>
static inline dtype subtract(dtype a, dtype b) {
    return a - b;
}

template <typename dtype>
static inline dtype multiply(dtype a, dtype b) {
    return a * b;
}

template <typename dtype>
static inline dtype divide(dtype a, dtype b) {
    if (b == 0) {
        // or return inf(-inf) ?
        throw std::invalid_argument("Division by zero.");
    }
    return a / b;
}

template <typename dtype>
static inline dtype power(dtype a, dtype b) {
    return std::pow(a, b);
}

// Overload operator<< to print Tensor
template <typename dtype>
std::ostream& operator<<(std::ostream& os, const Tensor<dtype>& tensor) {
    const auto& shape = tensor.shape();
    const auto& data = tensor.data();

    // scalar
    if (shape.size() == 0) {
        // os << "[]";
        os << tensor.data_[0];
    } else {
        tensor.printTensor(os, 0, {});
    }

    return os;
}

/**
 * same as operator<<, used for debug, since operator<< can not be called in vscode DEBUG CONSOLE.
 * @tparam dtype 
 */
template <typename dtype>
std::ostream& outputTensor(std::ostream& os, const Tensor<dtype>& tensor) {
    const auto& shape = tensor.shape();
    const auto& data = tensor.data();

    if (shape.size() == 0) {
        // os << "[]";
        os << tensor.data_[0];
    } else {
        tensor.printTensor(os, 0, {});
    }

    return os;
}

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

// Function to create a tensor filled with random numbers from a normal distribution
template <typename dtype>
Tensor<dtype> randn(const std::vector<int>& shape, dtype mean = 0.0, dtype stddev = 0.18) {
// Tensor<dtype> randn(const std::vector<int>& shape, dtype mean = 0.0, dtype stddev = 1.0) { // too big
    Tensor<dtype> result = Tensor<dtype>(shape);

    // Set up random number generation for normal distribution
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<dtype> distribution(mean, stddev);

    // Fill the tensor with random values
    for (int i = 0; i < result.num_elements; ++i) {
        result.data_[i] = distribution(generator);
    }
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
Tensor<dtype> apply_rotary_emb(Tensor<dtype> input, Tensor<dtype> freqs, int start_pos) {
    if (input.shape().size() != 4 || freqs.shape().size() != 2) {
        throw std::invalid_argument("Invalid shape.");
    }

    int B = input.shape()[0];
    int T = input.shape()[1];
    int n_heads = input.shape()[2];
    int head_dim = input.shape()[3];

    Tensor<dtype> result(input.shape());

    // #pragma omp parallel for collapse(4)
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < T; ++j) {
            for (int k = 0; k < n_heads; ++k) {
                for (int l = 0; l < head_dim; l += 2) {
                    // dtype theta = freqs.data_[j * freqs.shape()[0] + l / 2];
                    dtype theta = start_pos * 1.0f / std::pow(10000.0f, (float)l / (float)head_dim);
                    dtype cos_theta = std::cos(theta);
                    dtype sin_theta = std::sin(theta);

                    auto v0 = input.getData({i, j, k, l});
                    auto v1 = input.getData({i, j, k, l + 1});

                    result.setData({i, j, k, l}, v0 * cos_theta - v1 * sin_theta);
                    result.setData({i, j, k, l + 1}, v0 * sin_theta + v1 * cos_theta);
                }
            }
        }
    }

    return result;
    // return input;
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
inline const dtype& Tensor<dtype>::getData(const std::vector<int>& indices) const {
    size_t linear_index = calculateLinearIndex(indices);

    return data_[linear_index];
}


// Implementation of setData method
template <typename dtype>
inline void Tensor<dtype>::setData(const std::vector<int>& indices, const dtype& value) {
    size_t linear_index = calculateLinearIndex(indices);

    data_[linear_index] = value;
}
