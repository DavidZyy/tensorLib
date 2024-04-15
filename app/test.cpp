#include "../include/Tensor.hpp"
#include <cstddef>
#include "iostream"

int main() {
    Tensor<int> a({});
    for(size_t i=0; i<a.num_elements; i++)
        a.data_[i] = i;

    std::cout<<a;
}