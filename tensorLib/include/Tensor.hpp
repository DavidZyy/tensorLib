#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>
#include <ostream>
#include <atomic>

// Forward declaration of Tensor class
// template <typename dtype>
// class Tensor;

// Nested indexing using []
// tensor[i][j][k][t] return a tensor which shape 1, not a dtype.
// template <typename dtype>
// struct TensorProxy {
//     Tensor<dtype>& tensor;
//     std::vector<int> indices;
// 
//     // Constructor
//     TensorProxy(Tensor<dtype>& t, std::vector<int> idx) : tensor(t), indices(std::move(idx)) {}
// 
//     // Overloaded operator[] for nested indexing
//     TensorProxy operator[](int idx) {
//         indices.push_back(idx);
//         return *this;
//     }
// 
//     // Assignment operator to set value in the tensor
//     dtype& operator=(dtype value) {
//         tensor.setData(indices, value);
//         // size_t linear_index = calculateLinearIndex(indices);
//         return tensor(indices);
//     }
// };

static std::atomic<size_t> memoryUsage(0);

template <typename dtype>
class Deleter {
public:
    void operator()(dtype* ptr) const {
        memoryUsage -= sizeof(dtype) * elem_num;
        delete[] ptr;
        std::cout << "Free: " << sizeof(dtype) * elem_num << ", now: " << memoryUsage << std::endl;
    }

    Deleter(size_t elem_num) : elem_num(elem_num) {}

private:
    size_t elem_num;
};

template <typename dtype>
class Tensor {
public:
    // Constructor
    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, const std::shared_ptr<dtype[]>& data);
    Tensor(const std::vector<int>&& shape, const std::vector<int> &&stride, const int &offset, const std::shared_ptr<dtype[]>& data);

    // Destructor
    ~Tensor();

    // print methed // Declaration of friend function
    template <typename T>
    friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor);

    // Method to get shape
    const std::vector<int>& shape() const {
        return shape_;
    }

    // Method to get data (double is used as an example type)
    // const std::vector<dtype> data() const {
    const std::shared_ptr<dtype[]> data() const {
        return data_;
    }

    const dtype& getData(const std::vector<int>& indices) const;

    void setData(const std::vector<int>& indices, const dtype& value);

    // Accessor and modifier for tensor elements (non-const version)
    dtype& operator()(const std::vector<int>& indices);
    // Accessor for tensor elements (const version)
    const dtype& operator()(const std::vector<int>& indices) const;

    Tensor<int> operator==(const Tensor<dtype>& other) const;

    Tensor<dtype> operator*(const Tensor<dtype>& other) const;

    // Overloaded operator[] to return TensorProxy for nested indexing
    // need to return a new thensor with different shape_, stride_, offset_, ndim, but have the same data_ area.
    // TensorProxy<dtype> operator[](int idx) {
    //     return TensorProxy(*this, {idx});
    // }

    // Matrix multiplication method
    Tensor<dtype> matmul(const Tensor<dtype>& other) const;
    // Tensor<dtype> argmax(int axis) const;
    Tensor<int> argmax(int dim, bool keepdim = false) const;

    template<typename T = float>
    Tensor<T> mean(int dim = -1, bool keepdim = false) const;

    Tensor<dtype> view(const std::vector<int>& shape) const;

    // startIdx <= idx < endIdx
    Tensor<dtype> slice(int startIdx, int endIdx, int dim) const;

    /* Slices the input tensor along the selected dimension at the given index. 
        This function returns a view of the original tensor with the given dimension removed.*/
    Tensor<dtype> select(int dim, int index) const;

    dtype sum(bool keepdims = false) const;

    template<typename T>
    friend Tensor<T> maximum(Tensor<T> a, Tensor<T> b);

    template<typename T>
    friend Tensor<T> zeros(const std::vector<int>& shape);

    template<typename T>
    friend Tensor<T> apply_rotary_emb(Tensor<T> input, Tensor<T> freqs);

    Tensor<dtype> transpose(int dim0, int dim1) const;

    Tensor<dtype> permute(const std::vector<int>& dims) const;

    // return a new tensor with the same shape and data, 
    // but with a different memory layout which is contiguous.
    Tensor<dtype> contiguous() const;

    // get or set a sub-tensor of this tensor. The implementation here refers to the homework project of CMU10_414.
    Tensor<dtype> getItem(std::vector<std::vector<int>>& slices) const;
    void setItem(std::vector<std::vector<int>>& slices, const Tensor<dtype>& value);

    Tensor<dtype> broadcast_to(const std::vector<int>& new_shape) const;

    Tensor<dtype> exp();

    // reduce method(reduce 1 dimension) like sum
    Tensor<dtype> max();

    // int8_t quantize, but use int32_t store value now in case of overflow when perform mutmul.
    Tensor<int> quantize() const;

    Tensor<float> dequantize() const;

    /* data is managed by copy on write (COW) later */
    // std::vector<dtype> data_;
    std::shared_ptr<dtype[]> data_;
    int num_elements;

    // used for quantize
    // int group_size;  // seem as one group now for simple
    float scale;
private:
    // the offset of data_, used for slice method to share the same memory area of data_.
    int offset_;
    std::vector<int> stride_;
    int ndim;
    std::vector<int> shape_;

    // helper method for operator<<
    void printTensor(std::ostream& os, size_t depth, std::vector<int> indices) const;
    // Helper function to calculate linear index from multi-dimensional indices
    size_t calculateLinearIndex(const std::vector<int>& indices) const;
    // helper function for view
    bool is_contiguous(const Tensor<dtype>& t) const;
    // used in getItem method, process slice, supplement abbreviation of slice to full
    std::vector<std::vector<int>> process_slices(const std::vector<std::vector<int>>& slices) const;
};


// Overload operator<< to print Tensor
template <typename dtype>
std::ostream& operator<<(std::ostream& os, const Tensor<dtype>& tensor) {
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

template <typename dtype>
template <typename T>
// Tensor<float> Tensor<dtype>::mean(int dim, bool keepdim) const {
Tensor<T> Tensor<dtype>::mean(int dim, bool keepdim) const {
//     if (shape_.size() != 2) {
//         throw std::invalid_argument("Only support 2d.");
//     }
// 
//     int reduce_shape = shape_[1 - dim];
//     Tensor<int> result(std::vector<int>{reduce_shape});
// 
//     int off = stride_[1-dim];
//     int stride = stride_[dim];
// 
//     for (int i = 0; i < reduce_shape; ++i) {
//         int max_index = 0;
//         dtype max_value = data_[i*off];
//         for (int j = 0; j < shape_[dim]; ++j) {
//             if (data_[i*off + j*stride] > max_value) {
//                 max_value = data_[i*off + j*stride];
//                 max_index = j;
//             }
//         }
//         result.setData({i}, max_index);
//     }
// 
//     return result;
    
    if (shape_.size() != 1) {
        throw std::invalid_argument("Only support 1d.");
    }

    Tensor<T> result(std::vector<int>{1});

    dtype sum = 0;
    // for (auto value : data_) {
    //     sum += value;
    // }

    for(auto i = 0; i < this->num_elements; ++i) {
        sum += this->data_[i];
    }

    result.setData({0}, (T)sum / (T)shape_[0]);

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
Tensor<dtype> apply_rotary_emb(Tensor<dtype> input, Tensor<dtype> freqs) {
    if (input.shape().size() != 4 || freqs.shape().size() != 2) {
        throw std::invalid_argument("Invalid shape.");
    }

    int B = input.shape()[0];
    int T = input.shape()[1];
    int n_heads = input.shape()[2];
    int head_dim = input.shape()[3];

    Tensor<dtype> result(input.shape());

    #pragma omp parallel for collapse(4)
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < T; ++j) {
            for (int k = 0; k < n_heads; ++k) {
                for (int l = 0; l < head_dim; l += 2) {
                    dtype theta = freqs.data_[j * freqs.shape()[0] + l / 2];
                    dtype cos_theta = std::cos(theta);
                    dtype sin_theta = std::sin(theta);

                    auto v0 = input.getData({i, j, k, l});
                    auto v1 = input.getData({i, j, k, l + 1});

                    result.setData({i, j, k, l}, v0 * cos_theta - v1 * sin_theta);
                    result.setData({i, j, k, l + 1}, v0 * cos_theta + v1 * sin_theta);
                }
            }
        }
    }

    return result;
}
