#include "Tensor.hpp"
#include <cstddef>
#include <vector>
#include "iostream"
#include <chrono>

#include "Transformer.hpp"

// template<typename dtype>
// void rms_norm(dtype *output, dtype *input, dtype *weight, float epsilon, int hidden_size, int num_tokens);

int main() {
  int dim = 4;
  int tok = 4;
  // int dim = 4096;
  // int tok = 1024;

  // auto norm1 = RMSNorm<float> (dim, 1e-5, "cpu");
  // auto t = randn <float> ({tok, dim}, "cpu");

  auto norm1 = nn::RMSNorm<float> (dim, 1e-5, "cuda");
  auto t = randn <float> ({tok, dim}, "cuda");

  // calcualte the forward pass time 
  auto start = std::chrono::high_resolution_clock::now();
  auto r1 = norm1.forward(t);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken for forward pass: " << elapsed.count() << " s\n";

  start = std::chrono::high_resolution_clock::now();
  auto r2 = norm1.forward_fused_cuda(t);
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_fused = end - start;
  std::cout << "Time taken for fused forward pass: " << elapsed_fused.count() << " s\n";

  // std::cout << "t: " << t << std::endl;
  // std::cout << "r1: " << norm1.forward(t) << std::endl;
  // std::cout << "r2: " << norm1.forward_fused_cuda(t) << std::endl;
  std::cout << "r1: " << r1 << std::endl;
  std::cout << "r2: " << r2 << std::endl;
}
