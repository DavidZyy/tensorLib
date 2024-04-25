#include "../../include/Tensor.hpp"
#include "../../include/nn/modules.hpp"
#include <iostream>

Tensor<int> originTensor(const std::vector<int>& shape) {
    Tensor<int> tensor(shape);

    for(auto i=0; i<tensor.num_elements; i++)
        tensor.data_[i] = i;

    return tensor;
}

void test_ReLU() {
    Tensor<int> a = originTensor({2, 3, 4});
    nn::ReLU<int> relu;
    Tensor<int> b = relu.forward(a);

    std::cout << "a: " << std::endl << a << std::endl;
    std::cout << "b: " << std::endl << b << std::endl;
}

void test_Conv2d() {
    // N * C * H * W
    Tensor<int> input = originTensor({1,2, 5, 5});
    // c_cout * c_in * kernel_size * kernel_size
    Tensor<int> weight = originTensor({3, 2, 3, 3});

    nn::Conv2d<int> conv2d(2, 3, 3, 1, 0, std::move(weight));
    Tensor<int> output = conv2d.forward(input);

    std::cout << "input: "  << std::endl << input  << std::endl;
    std::cout << "weight: "  << std::endl << weight  << std::endl;
    std::cout << "output: " << std::endl << output << std::endl;
}

int main() {
    // test_ReLU();
    test_Conv2d();
    return 0;
}
