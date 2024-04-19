#include "../../include/Tensor.hpp"
#include "../../include/nn/modules.hpp"
#include <iostream>

Tensor<int> originTensor(const std::vector<int>& shape) {
    Tensor<int> tensor(shape);

    for(auto i=0; i<tensor.num_elements; i++)
        tensor.data_[i] = i - 8;

    return tensor;
}

void test_ReLU() {
    Tensor<int> a = originTensor({2, 3, 4});
    nn::ReLU<int> relu;
    Tensor<int> b = relu.forward(a);

    std::cout << "a: " << std::endl << a << std::endl;
    std::cout << "b: " << std::endl << b << std::endl;
}

int main() {
    test_ReLU();
    return 0;
}
