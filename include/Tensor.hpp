#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <ostream>

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


    std::vector<dtype> data_;
    int num_elements;

private:
    std::unique_ptr<int[]> offset_;
    std::unique_ptr<int[]> stride_;
    int ndim;
    std::vector<int> shape_;

    // helper method for operator<<
    void printTensor(std::ostream& os, size_t depth, std::vector<int> indices) const;

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
