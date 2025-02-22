#include "nn/modules.hpp"

namespace nn {

template class Module<half>;
template class Module<float>;
template class Module<int>;

// not pure virtual, should have defination
template <typename dtype>
// Tensor<dtype> Module<dtype>::forward(const Tensor<dtype> &input, std::optional<Tensor<dtype>> result_opt) const {
Tensor<dtype> Module<dtype>::forward(const Tensor<dtype> &input) const {
  throw std::runtime_error("module's forward should never be called!");
  return input;
}

}
