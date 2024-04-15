#pragma once

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

    std::vector<dtype> data_;
    int num_elements;
private:
    std::unique_ptr<int[]> offset_;
    std::unique_ptr<int[]> stride_;
    int ndim;
    std::vector<int> shape_;
};

 // Overload operator<< to print Tensor
template <typename dtype>
std::ostream& operator<<(std::ostream& os, const Tensor<dtype>& tensor) {
    const auto& shape = tensor.shape();
    const auto& data = tensor.data();

    if (shape.size() == 0) {
        os << "[]";
    } else if (shape.size() == 1) {
        // 1D Tensor
        os << "[";
        for (size_t i = 0; i < shape[0]; ++i) {
            if (i > 0) os << ", ";
            os << data[i];
        }
        os << "]";
    } else if (shape.size() == 2) {
        // 2D Tensor
        os << "[";
        for (size_t i = 0; i < shape[0]; ++i) {
            if (i > 0) os << std::endl << " ";
            os << "[";
            for (size_t j = 0; j < shape[1]; ++j) {
                if (j > 0) os << ", ";
                os << data[i * shape[1] + j];
            }
            os << "]";
        }
        os << "]";
    } else if (shape.size() == 3) {
        // 3D Tensor
        os << "[";
        for (size_t i = 0; i < shape[0]; ++i) {
            if (i > 0) os << std::endl << " ";
            os << "[";
            for (size_t j = 0; j < shape[1]; ++j) {
                if (j > 0) os << ", ";
                os << "[";
                for (size_t k = 0; k < shape[2]; ++k) {
                    if (k > 0) os << ", ";
                    os << data[(i * shape[1] + j) * shape[2] + k];
                }
                os << "]";
            }
            os << "]";
        }
        os << "]";
    } else if (shape.size() == 4) {
        // 4D Tensor
        os << "[";
        for (size_t i = 0; i < shape[0]; ++i) {
            if (i > 0) os << std::endl << " ";
            os << "[";
            for (size_t j = 0; j < shape[1]; ++j) {
                if (j > 0) os << std::endl << "  ";
                os << "[";
                for (size_t k = 0; k < shape[2]; ++k) {
                    if (k > 0) os << ", ";
                    os << "[";
                    for (size_t l = 0; l < shape[3]; ++l) {
                        if (l > 0) os << ", ";
                        os << data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
                    }
                    os << "]";
                }
                os << "]";
            }
            os << "]";
        }
        os << "]";
    } else {
        // Unsupported higher dimensions
        os << "Unsupported dimension";
    }
    os<<std::endl;

    return os;
}
