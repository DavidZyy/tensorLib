#pragma once

#include "Tensor.hpp"
#include "iostream"
#include <cassert>
#include <chrono>
#include <vector>

namespace nn {

// template class Module<float>;
template <typename dtype> class Module {
public:
  Module() = default;
  virtual ~Module() = default;
  // virtual Tensor<dtype> forward(const Tensor<dtype>& intput) const = 0; //
  // pure virtual func
  virtual Tensor<dtype> forward(const Tensor<dtype> &intput)
      const; // not pure, for overloading forward(that have different numbers or
             // types of parameters)
};

// not pure virtual, should have defination
template <typename dtype>
Tensor<dtype> Module<dtype>::forward(const Tensor<dtype> &input) const {
  throw std::runtime_error("module's forward should never be called!");
  return input;
}

template <typename dtype> class Linear : public Module<dtype> {
public:
  Linear() = default; // used in member initializer list.
  Linear(int in_features, int out_features);
  Linear(int in_features, int out_features, Tensor<dtype> &&weight);
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

template <typename dtype>
Linear<dtype>::Linear(int in_features, int out_features)
    : in_features(in_features), out_features(out_features),
      weight(Tensor<dtype>(std::vector<int>{out_features, in_features})){
      // weight(randn<dtype>({out_features, in_features})) { // randn use a lot of time when parameter initialization
} // maybe should add kaiming uniform first !!

// template <typename dtype>
// Linear<dtype>::Linear(int in_features, int out_features) :
// in_features(in_features), out_features(out_features) {
//     weight = Tensor<dtype>(std::vector<int>{out_features, in_features});
// }

template <typename dtype>
Linear<dtype>::Linear(int in_features, int out_features, Tensor<dtype> &&weight)
    : in_features(in_features), out_features(out_features),
      weight(std::move(weight)) {
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

template <typename dtype>
// class ReLU : public Module<dtype> {
class ReLU : public nn::Module<dtype> {
public:
  ReLU() = default;
  ~ReLU() = default;
  Tensor<dtype> forward(const Tensor<dtype> &input); // maximum method
};

template <typename dtype>
Tensor<dtype> ReLU<dtype>::forward(const Tensor<dtype> &input) {
  Tensor<dtype> temp({});
  temp.setData({}, 0);
  return maximum(input, temp);
}

template <typename dtype>
// class Conv2d : public Module<dtype> {
class Conv2d : public nn::Module<dtype> {
public:
  Conv2d() = default;
  Conv2d(int in_channels, int out_channels, int kernel_size, int stride,
         int padding, Tensor<dtype> &&weight);
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

template <typename dtype>
Conv2d<dtype>::Conv2d(int in_channels, int out_channels, int kernel_size,
                      int stride, int padding, Tensor<dtype> &&weight)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_size(kernel_size), stride(stride), padding(padding),
      weight(std::move(weight)) {

  // get error when construct a conv layer.
  // weight = Tensor<dtype>({in_channels, kernel_size, kernel_size,
  // out_channels}); Tensor<dtype> a({in_channels, kernel_size, kernel_size,
  // out_channels}); weight({in_channels, kernel_size, kernel_size,
  // out_channels});
}

/**
 * input shape:  N x c_in x H x W
 * weight shape: c_cout * c_in * kernel_size * kernel_size
 * output shape: N x c_cout x H_out x W_out
 */
template <typename dtype>
Tensor<dtype> Conv2d<dtype>::forward(const Tensor<dtype> &input) {
  auto start_time = std::chrono::high_resolution_clock::now();

  assert(input.shape().size() == 4 && input.shape()[1] == in_channels);

  auto output_height =
      (input.shape()[2] + 2 * padding - kernel_size) / stride + 1;
  auto output_width =
      (input.shape()[3] + 2 * padding - kernel_size) / stride + 1;
  auto output_shape = std::vector<int>{input.shape()[0], out_channels,
                                       output_height, output_width};

  // padding
  auto input_padded = zeros<dtype>({input.shape()[0], input.shape()[1],
                                    input.shape()[2] + 2 * padding,
                                    input.shape()[3] + 2 * padding});
  for (int i = 0; i < input.shape()[0]; i++) {
    for (int j = 0; j < input.shape()[1]; j++) {
      for (int k = 0; k < input.shape()[2]; k++) {
        for (int t = 0; t < input.shape()[3]; t++) {
          input_padded.setData({i, j, k + padding, t + padding},
                               input.getData({i, j, k, t}));
        }
      }
    }
  }

  auto output = Tensor<dtype>(output_shape);

  // auto start_time = std::chrono::high_resolution_clock::now();
  // conv
  for (int idxn = 0; idxn < output_shape[0]; idxn++) {
    // printf("idxn: %d\n", idxn);
    for (int idxc = 0; idxc < output_shape[1]; idxc++) {
      for (int idxh = 0; idxh < output_shape[2]; idxh++) {
        for (int idxw = 0; idxw < output_shape[3]; idxw++) {
          // weight select channel idxc, input_padded select data idxn.
          auto convTensor =
              weight.select(0, idxc) *
              input_padded.select(0, idxn)
                  .slice(idxh * stride, idxh * stride + kernel_size, 1)
                  .slice(idxw * stride, idxw * stride + kernel_size, 2);
          output.setData({idxn, idxc, idxh, idxw}, convTensor.sum());
        }
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                start_time)
          .count();
  std::cout << "Conv Execution time: " << duration_seconds << " seconds"
            << std::endl;

  return output;
}

template <typename dtype> class Embedding : public Module<dtype> {
public:
  Embedding() = default;
  Embedding(int num_embeddings, int embedding_dim);
  ~Embedding() = default;

  Tensor<dtype> forward(const Tensor<dtype> &input) const override;

  // private:
  // protected:
  int num_embeddings;
  int embedding_dim;
  Tensor<dtype> weight;
};

template <typename dtype>
Embedding<dtype>::Embedding(int num_embeddings, int embedding_dim)
    : num_embeddings(num_embeddings), embedding_dim(embedding_dim),
      weight(Tensor<dtype>({num_embeddings, embedding_dim})) {}
      // weight(randn<dtype>({num_embeddings, embedding_dim})) {}
// num_embeddings(num_embeddings), embedding_dim(embedding_dim),
// weight(Tensor<dtype>({num_embeddings, embedding_dim})) {}

/**
 * using the following way, we can handle arbitrary dimension input.
 * @tparam dtype
 */
template <typename dtype>
Tensor<dtype> Embedding<dtype>::forward(const Tensor<dtype> &input) const {
  auto new_shape = input.shape();
  new_shape.push_back(embedding_dim);
  auto result = Tensor<dtype>(new_shape);

  std::vector<int> cur_idx(input.shape().size(), 0);

  // embedding every elements in input
  for (int i = 0; i < input.num_elements; i++) {
    int embedding_index = input.getData(cur_idx);

    // assign
    for (int j = 0; j < embedding_dim; j++) {
      auto temp = cur_idx;
      temp.push_back(j);
      result.setData(temp, weight.getData({embedding_index, j}));
    }

    // carry
    // maybe optimized
    for (int j = cur_idx.size() - 1; j >= 0; j--) {
      cur_idx[j] += 1;

      if (cur_idx[j] < input.shape()[j]) {
        break;
      } else {
        cur_idx[j] = 0;
      }
    }
  }

  return result;
}

// Define the ModuleList class
template <typename dtype> class ModuleList : public Module<dtype> {
public:
  // Default constructor
  ModuleList() = default;

  // Append a new module to the list
  void append(std::shared_ptr<Module<dtype>> module) {
    modules_.push_back(module);
  }

  // Access a module by index
  std::shared_ptr<Module<dtype>> operator[](int index) const {
    if (index < 0 || index >= modules_.size()) {
      throw std::out_of_range("ModuleList index out of range.");
    }
    return modules_[index];
  }

  // Get the size of the ModuleList
  int size() const { return modules_.size(); }

  // Forward pass through all modules in the list
  Tensor<dtype> forward(const Tensor<dtype> &input) const {
    Tensor<dtype> current_input = input;

    // Sequentially pass through all modules in the list
    for (const auto &module : modules_) {
      current_input = module->forward(current_input);
    }

    return current_input;
  }

private:
  // Vector to hold the list of shared pointers to modules
  std::vector<std::shared_ptr<Module<dtype>>> modules_;
};

} // namespace nn