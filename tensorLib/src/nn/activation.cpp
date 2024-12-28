/**
 * @file activation.cpp
 * @author Yangyang Zhu (yangyangzhu12@qq.com)
 * @version 0.1
 * @date 2024-12-28
 * activation functions, including ReLU, Sigmoid, Tanh, Softmax ...
 * 
 */

#include "nn/activation.hpp"

template <typename dtype>
Tensor<dtype> ReLU<dtype>::forward(const Tensor<dtype> &input) {
  Tensor<dtype> temp({});
  temp.setData({}, 0);
  return maximum(input, temp);
}
