#pragma once

#include "../Tensor.hpp"
#include <cassert>

namespace nn {

template <typename dtype>
class Linear {
public:
    Linear(int in_features, int out_features);
    Linear(int in_features, int out_features, Tensor<dtype>&& weight);
    ~Linear() = default;
    Tensor<dtype> forward(const Tensor<dtype>& input);

protected:
    int in_features;
    int out_features;
    Tensor<dtype> weight;
};

template <typename dtype>
Linear<dtype>::Linear(int in_features, int out_features) : in_features(in_features), out_features(out_features) {
    weight = Tensor<dtype>(std::vector<int>{out_features, in_features});
}

template <typename dtype>
Linear<dtype>::Linear(int in_features, int out_features, Tensor<dtype>&& weight)
        : in_features(in_features), out_features(out_features), weight(std::move(weight)) {
    // Optionally perform some sanity checks on the weight tensor shape
    // assert(weight.shape().size() == 2 && weight.shape()[0] == out_features && weight.shape()[1] == in_features);
    assert(weight.shape().size() == 2 && weight.shape()[1] == out_features && weight.shape()[0] == in_features);
}

template <typename dtype>
Tensor<dtype> Linear<dtype>::forward(const Tensor<dtype>& input) {
    return input.matmul(weight);
}


template <typename dtype>
class ReLU {
public:
    ReLU() = default;
    ~ReLU() = default;
    Tensor<dtype> forward(const Tensor<dtype>& input); // maximum method
};

template <typename dtype>
Tensor<dtype> ReLU<dtype>::forward(const Tensor<dtype>& input) {
    Tensor<int> temp({});
    temp.setData({}, 0);
    return maximum(input, temp);
}

}
