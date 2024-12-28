#include "nn/conv.hpp"
#include <chrono>

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

