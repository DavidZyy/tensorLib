#include "nn/linear.hpp"

namespace nn {

template class Linear<float>; // if not add this, will get error: undefined reference to `nn::Linear<float>::Linear(int, int, std::string)'
template class Linear<int>;

template <typename dtype>
Linear<dtype>::Linear(int in_features, int out_features, std::string device_type)
    : Module<dtype>(device_type), in_features(in_features), out_features(out_features),
      weight(Tensor<dtype>(std::vector<int>{out_features, in_features}, device_type)){
      // weight(randn<dtype>({out_features, in_features})) { // randn use a lot of time when parameter initialization
} // maybe should add kaiming uniform first !!

// template <typename dtype>
// Linear<dtype>::Linear(int in_features, int out_features) :
// in_features(in_features), out_features(out_features) {
//     weight = Tensor<dtype>(std::vector<int>{out_features, in_features});
// }

template <typename dtype>
Linear<dtype>::Linear(int in_features, int out_features, Tensor<dtype> &&weight, std::string device_type)
    : in_features(in_features), out_features(out_features),
      weight(std::move(weight)) {
  assert(weight.device_type == device_type);
  // Optionally perform some sanity checks on the weight tensor shape
  assert(weight.shape().size() == 2 && weight.shape()[0] == out_features &&
         weight.shape()[1] == in_features);
  // assert(weight.shape().size() == 2 && weight.shape()[1] == out_features &&
  // weight.shape()[0] == in_features);
}

/**
 * input:  (N, in_features)
 * weight: (out_features, in_features)
 * output: (N, out_features)
 * output = input.matmul(weight.T)
 */
template <typename dtype>
Tensor<dtype> Linear<dtype>::forward(const Tensor<dtype> &input) const {
  // auto start_time = std::chrono::high_resolution_clock::now();

  auto result = input.matmul(weight.transpose(0, 1));

  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration_seconds =
  // std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
  // start_time).count(); std::cout << "Linear Execution time: " <<
  // duration_seconds << " seconds" << std::endl;

  return result;
}

}   // namespace nn 
