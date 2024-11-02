#include "Tensor.hpp"
#include <ostream>
#include <chrono>

int main() {
    Tensor<float> a = full({10000, 10000}, 1.0f, "cpu");
    Tensor<float> b = full({10000,10000}, 1.0f, "cuda");

    // std::cout << "a: " << std::endl << a << std::endl;
    // std::cout << "b: " << std::endl << b << std::endl;

    // Measure the time for CPU matmul
    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto c = a.matmul(a);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU matmul time: " << cpu_duration.count() << " seconds" << std::endl;

    // Measure the time for CUDA matmul
    auto start_cuda = std::chrono::high_resolution_clock::now();
    auto d = b.matmul(b);
    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cuda_duration = end_cuda - start_cuda;
    std::cout << "CUDA matmul time: " << cuda_duration.count() << " seconds" << std::endl;

    // std::cout << "c: " << std::endl << c << std::endl;
    // std::cout << "d: " << std::endl << d << std::endl;
}
