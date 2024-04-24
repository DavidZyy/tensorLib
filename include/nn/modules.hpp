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

template <typename dtype>
class Conv2d {
public:
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, Tensor<dtype>&& weight);
    ~Conv2d() = default;

    Tensor<dtype> forward(const Tensor<dtype>& input);

// private:
protected:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    Tensor<dtype> weight;
};

template <typename dtype>
Conv2d<dtype>::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, Tensor<dtype>&& weight) : 
    in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding), weight(std::move(weight)) {
    
    // get error when construct a conv layer.
    // weight = Tensor<dtype>({in_channels, kernel_size, kernel_size, out_channels});
    // Tensor<dtype> a({in_channels, kernel_size, kernel_size, out_channels});
    // weight({in_channels, kernel_size, kernel_size, out_channels});
}

/**
 * input shape: N x C x H x W 
 * need Tensor op: slice, elem_mul, sum
 * @tparam dtype 
 */
template <typename dtype>
Tensor<dtype> Conv2d<dtype>::forward(const Tensor<dtype>& input) {
    assert(input.shape().size() == 4 && input.shape()[1] == in_channels);

    auto output_height = (input.shape()[2] + 2 * padding - kernel_size) / stride + 1;
    auto output_width = (input.shape()[3] + 2 * padding - kernel_size) / stride + 1;
    auto output_shape = std::vector<int>{input.shape()[0], out_channels, output_height, output_width};

    auto input_padded = zeros<dtype>({input.shape()[0], input.shape()[1], input.shape()[2] + 2 * padding, input.shape()[3] + 2 * padding});
    for (int i = 0; i < input.shape()[0]; i++) {
        for (int j = 0; j < input.shape()[1]; j++) {
            for (int k = 0; k < input.shape()[2]; k++) {
                for (int t = 0; t < input.shape()[3]; t++) {
                    input_padded.setData({i, j, k + padding, t + padding}, input.getData({i, j, k, t}));
                }
            }
        }
    }

    auto output = Tensor<dtype>(output_shape);

    // conv
    for (int idxn = 0; idxn < output_shape[0]; idxn++) {
        for (int idxc = 0; idxc < output_shape[1]; idxc++) {
            for (int idxh = 0; idxh < output_shape[2]; idxh++) {
                for (int idxw = 0; idxw < output_shape[3]; idxw++) {
                    // weight select channel idxc, input_padded select data idxn.
                    auto convTensor = weight.slice(idxc, idxc+1, 3) * input_padded.slice(idxn, idxn+1, 0).slice(idxh*stride, idxh*stride+kernel_size, 2).slice(idxw*stride, idxw*stride+kernel_size, 3);
                    output.setData({idxn, idxc, idxh, idxw}, convTensor.sum());
                }
            }
        }
    }

    return output;
}

}
