#include "../include/Tensor.hpp"
#include <vector>

// Explicit instantiation for int
template class Tensor<int>;

// Explicit instantiation for double
template class Tensor<double>;

template <typename dtype>
Tensor<dtype>::Tensor(const std::vector<int>& shape) : ndim(shape.size()), shape_(shape) {
        // Calculate the total number of elements in the tensor
        if(shape.empty()) {
            num_elements = 0;
        } else {
            num_elements = 1;
            for (int dim : shape) {
                num_elements *= dim;
            }
        }

        // Allocate memory for data, offset, and stride arrays
        // data_ = std::make_unique<double[]>(num_elements);
        data_ = std::vector<dtype>(num_elements);
        offset_ = std::make_unique<int[]>(ndim);
        stride_ = std::make_unique<int[]>(ndim);

        // Initialize offset and stride arrays
        if(ndim > 0) {
            // offset_[ndim - 1] = 1;
            stride_[ndim - 1] = 1;
            for (int i = ndim - 2; i >= 0; --i) {
                // offset_[i] = offset_[i + 1] * shape_[i + 1];
                stride_[i] = stride_[i + 1] * shape_[i + 1];
            }
        }
}

template <typename dtype>
Tensor<dtype>::~Tensor() {
    // Implement destructor logic here
    // Automatically release allocated memory
}

