#pragma once
#include "nn/modules.hpp"

template <typename dtype>
// class ReLU : public Module<dtype> {
class ReLU : public nn::Module<dtype> {
public:
  ReLU() = default;
  ~ReLU() = default;
  Tensor<dtype> forward(const Tensor<dtype> &input); // maximum method

};
