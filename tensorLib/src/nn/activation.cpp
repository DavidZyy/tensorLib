#include "nn/activation.hpp"

template <typename dtype>
Tensor<dtype> ReLU<dtype>::forward(const Tensor<dtype> &input) {
  Tensor<dtype> temp({});
  temp.setData({}, 0);
  return maximum(input, temp);
}
