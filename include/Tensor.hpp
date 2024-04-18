#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>
#include <ostream>

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

template <typename dtype>
class Tensor {
public:
    // Constructor
    Tensor(const std::vector<int>& shape);
    // the data may be another's Tensor's data, allocated in heap, or a temporary vector, in stack,
    // maybe cause error.
    Tensor(const std::vector<int>& shape, const std::vector<dtype>& data);

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
    const std::vector<dtype> data() const {
        return data_;
    }

    const dtype& getData(const std::vector<int>& indices) const;

    void setData(const std::vector<int>& indices, const dtype& value);

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

    std::vector<dtype> data_;
    int num_elements;

    // Matrix multiplication method
    Tensor<dtype> matmul(const Tensor<dtype>& other) const;
    // Tensor<dtype> argmax(int axis) const;
    Tensor<int> argmax(int dim, bool keepdim = false) const;

    template<typename T = float>
    Tensor<T> mean(int dim = -1, bool keepdim = false) const;

    Tensor<dtype> view(const std::vector<int>& shape) const;

private:
    std::vector<int> offset_;
    std::vector<int> stride_;
    int ndim;
    std::vector<int> shape_;

    // helper method for operator<<
    void printTensor(std::ostream& os, size_t depth, std::vector<int> indices) const;
    // Helper function to calculate linear index from multi-dimensional indices
    size_t calculateLinearIndex(const std::vector<int>& indices) const;
    // helper function for view
    bool is_contiguous(const Tensor<dtype>& t) const;
};


// Overload operator<< to print Tensor
template <typename dtype>
std::ostream& operator<<(std::ostream& os, const Tensor<dtype>& tensor) {
    const auto& shape = tensor.shape();
    const auto& data = tensor.data();

    if (shape.size() == 0) {
        os << "[]";
    } else {
        tensor.printTensor(os, 0, {});
    }

    return os;
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
    for (auto value : data_) {
        sum += value;
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

