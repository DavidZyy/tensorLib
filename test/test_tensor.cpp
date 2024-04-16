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


// Function to test the getData method of the Tensor class
int test_getData() {
    // Define a shape for the tensor (e.g., 2x3x4)
    // std::vector<int> shape = {2, 3, 4};
    std::vector<int> shape = {2, 3, 4};

    // Create a tensor with the specified shape
    Tensor<int> tensor(shape);

    // Fill the tensor with some test data (e.g., sequential numbers)
    int value = 0;
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                tensor.data_[i * shape[1] * shape[2] + j * shape[2] + k] = value++;
            }
        }
    }

    // Test the getData method by retrieving specific elements and checking the values
    // Expected values: tensor.getData({0, 0, 0}) = 0, tensor.getData({1, 2, 3}) = 23, etc.
    int expected_value = 0;
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                std::vector<int> indices = {i, j, k};
                int actual_value = tensor.getData(indices);
                if (actual_value != expected_value) {
                    std::cerr << "Error: Incorrect value retrieved at indices "
                              << "[" << i << "][" << j << "][" << k << "]. "
                              << "Expected: " << expected_value << ", Actual: " << actual_value << std::endl;
                    return 1; // Return failure
                }
                expected_value++;
            }
        }
    }

    std::cout << "getData method test passed!" << std::endl;
    return 0; // Return success
}

void test_setData() {
    // Create a Tensor object with a specific shape
    std::vector<int> shape = {5, 4, 3, 2};
    // std::vector<int> shape = {};
    Tensor<int> tensor(shape);

    // Set data at specific indices
    // tensor.setData({0, 0}, 1);
    // tensor.setData({0, 1}, 2);
    // tensor.setData({0, 2}, 3);
    // tensor.setData({1, 0}, 4);
    // tensor.setData({1, 1}, 5);
    // tensor.setData({1, 2}, 6);

    for(auto i=0; i<tensor.num_elements; i++)
        tensor.data_[i] = i;

    // Print the tensor to verify the data
    std::cout << "Tensor after setData:" << std::endl;
    std::cout << tensor << std::endl;
}

void test_parenthesis() {
    // Create a Tensor object with a specific shape
    std::vector<int> shape = {5, 4, 3, 2};
    Tensor<int> tensor(shape);

    int value = 0;
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                for(int t = 0; t < shape[3]; ++t) {
                    tensor({i, j, k, t}) = value++;
                }
            }
        }
    }

    std::cout << "Tensor after setData:" << std::endl;
    std::cout << tensor << std::endl;
}

// void test_square_brackets() {
//     // Create a Tensor object with a specific shape
//     std::vector<int> shape = {5, 4, 3, 2};
//     Tensor<int> tensor(shape);
// 
//     int value = 0;
//     for (int i = 0; i < shape[0]; ++i) {
//         for (int j = 0; j < shape[1]; ++j) {
//             for (int k = 0; k < shape[2]; ++k) {
//                 for(int t = 0; t < shape[3]; ++t) {
//                     // tensor({i, j, k, t}) = value++;
//                     tensor[i][j][k][t] = value++;
//                 }
//             }
//         }
//     }
//     auto a = tensor[1][2][3][0];
// 
//     std::cout << "Tensor after setData:" << std::endl;
//     std::cout << tensor << std::endl;
//     // std::cout << tensor[1][2][3][0] << std::endl;
// }

int main() {
    test_construct();
    // test_getData();
    // test_setData();
    // test_parenthesis();
    // test_square_brackets();
}
