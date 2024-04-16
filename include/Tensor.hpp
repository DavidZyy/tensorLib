#pragma once

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

    // Overloaded operator[] to return TensorProxy for nested indexing
    // need to return a new thensor with different shape_, stride_, offset_, ndim, but have the same data_ area.
    // TensorProxy<dtype> operator[](int idx) {
    //     return TensorProxy(*this, {idx});
    // }

    std::vector<dtype> data_;
    int num_elements;

private:
    std::vector<int> offset_;
    std::vector<int> stride_;
    int ndim;
    std::vector<int> shape_;

    // helper method for operator<<
    void printTensor(std::ostream& os, size_t depth, std::vector<int> indices) const;
    // Helper function to calculate linear index from multi-dimensional indices
    size_t calculateLinearIndex(const std::vector<int>& indices) const;
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
