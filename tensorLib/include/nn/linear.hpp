#pragma once
#include "nn/modules.hpp"

namespace nn {

template <typename dtype> 
class Linear : public Module<dtype> {
public:
  Linear() = default; // used in member initializer list.
  Linear(int in_features, int out_features, std::string device_type = "cpu");
  Linear(int in_features, int out_features, Tensor<dtype> &&weight, std::string device_type = "cpu");
  ~Linear() = default;
  Tensor<dtype> forward(const Tensor<dtype> &input)
      const override; // add const will resolve the bug, but why??(because this
                      // virtual function has const at last, the inherited class
                      // should have const too)

  // protected:
  int in_features;
  int out_features;
  Tensor<dtype> weight;
};

} // namespace nn
