#pragma once
#include "Tensor.hpp"

namespace nn {

// template class Module<float>;
template <typename dtype> 
class Module {
public:
  Module() = default;
  Module(std::string device_type) : device_type(device_type) {}
  virtual ~Module() = default;
  // virtual Tensor<dtype> forward(const Tensor<dtype>& intput) const = 0; //
  // pure virtual func
  virtual Tensor<dtype> forward(const Tensor<dtype> &intput)
      const; // not pure, for overloading forward(that have different numbers or
             // types of parameters)

// private:
  std::string device_type;
};

} // namespace nn
