#pragma once
#include "nn/modules.hpp"

template <typename dtype>
// class Conv2d : public Module<dtype> {
class Conv2d : public nn::Module<dtype> {
public:
  Conv2d() = default;
  Conv2d(int in_channels, int out_channels, int kernel_size, int stride,
         int padding, Tensor<dtype> &&weight, std::string device_type = "cpu");
  ~Conv2d() = default;

  Tensor<dtype> forward(const Tensor<dtype> &input);

  // private:
protected:
  int in_channels;
  int out_channels;
  int kernel_size;
  int stride;
  int padding;
  // c_cout * c_in * kernel_size * kernel_size
  Tensor<dtype> weight;
};

