#include "../include/Tensor.hpp"
#include <cstddef>
#include <vector>

// Explicit instantiation for int
template class Tensor<int>;

// Explicit instantiation for double
template class Tensor<double>;

// Explicit instantiation for float
template class Tensor<float>;

template class Tensor<uint8_t>;

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
        data_ = std::vector<dtype>(num_elements);
        offset_ = std::vector<int>(ndim);
        stride_ = std::vector<int>(ndim);

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

}

template <typename dtype>
size_t Tensor<dtype>::calculateLinearIndex(const std::vector<int>& indices) const{
    // doulble check
    if (indices.size() != shape_.size() || indices.size() != ndim) {
        throw std::invalid_argument("Error: Indices size does not match tensor dimension");
    }

    size_t linear_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range("Error: Index out of range");
        }
        linear_index += indices[i] * stride_[i];
    }

    return linear_index;
}

template <typename dtype>
const dtype& Tensor<dtype>::getData(const std::vector<int>& indices) const {
    size_t linear_index = calculateLinearIndex(indices);

    return data_[linear_index];
}


// Implementation of setData method
template <typename dtype>
void Tensor<dtype>::setData(const std::vector<int>& indices, const dtype& value) {
    size_t linear_index = calculateLinearIndex(indices);

    data_[linear_index] = value;
}

// Accessor implementation (non-const version)
template <typename dtype>
dtype& Tensor<dtype>::operator()(const std::vector<int>& indices) {
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
            os << data_[idx + i];
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
            indices.pop_back();
        }
        os << "]";
    }
}

/**
 * Matrix multiplication method implementation
 */
template <typename dtype>
Tensor<dtype> Tensor<dtype>::matmul(const Tensor<dtype>& other) const {
    // Check dimensions for compatibility
    if (shape_.size() != 2 || other.shape().size() != 2 || shape_[1] != other.shape()[0]) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication");
    }

    // Dimensions of the resulting matrix
    std::vector<int> result_shape = {shape_[0], other.shape()[1]};
    Tensor<dtype> result(result_shape);

    // Perform matrix multiplication
    for (int i = 0; i < shape_[0]; ++i) {
        for (int j = 0; j < other.shape()[1]; ++j) {
            dtype sum = 0;
            for (int k = 0; k < shape_[1]; ++k) {
                sum += this->getData({i, k}) * other.getData({k, j});
            }
            result.setData({i, j}, sum);
        }
    }

    return result;
}

/**
 * Returns the indices of the maximum values along an axis.
 * @param dim the dimension to reduce.
 */
template <typename dtype>
Tensor<int> Tensor<dtype>::argmax(int dim, bool keepdim) const{
    if (shape_.size() != 2) {
        throw std::invalid_argument("Only support 2d.");
    }

    int reduce_shape = shape_[1 - dim];
    Tensor<int> result(std::vector<int>{reduce_shape});

    int off = stride_[1-dim];
    int stride = stride_[dim];

    for (int i = 0; i < reduce_shape; ++i) {
        int max_index = 0;
        dtype max_value = data_[i*off];
        for (int j = 0; j < shape_[dim]; ++j) {
            if (data_[i*off + j*stride] > max_value) {
                max_value = data_[i*off + j*stride];
                max_index = j;
            }
        }
        result.setData({i}, max_index);
    }

    return result;
}
