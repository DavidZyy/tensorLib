#include "../include/Tensor.hpp"
#include <cstddef>
#include "iostream"

int test_construct() {
    // Define a list of shapes for the tensors
    std::vector<std::vector<int>> shapes = {
        {},
        {1},
        {2, 3},        // 2D tensor with shape (2, 3)
        {3, 4, 2},     // 3D tensor with shape (3, 4, 2)
        {2, 2, 2, 2}   // 4D tensor with shape (2, 2, 2, 2)
    };

    // Create and manipulate tensors with different shapes
    for (const auto& shape : shapes) {
        Tensor<int> tensor(shape);

        // Assign values to the tensor
        for (size_t i = 0; i < tensor.num_elements; ++i) {
            tensor.data_[i] = static_cast<int>(i);
        }

        // Print the tensor
        std::cout << "Tensor with shape ";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) {
                std::cout << " x ";
            }
        }
        std::cout << ":" << std::endl;
        std::cout << tensor << std::endl;
        std::cout << std::endl;
    }

    return 0;
}

int main() {
    test_construct();
}