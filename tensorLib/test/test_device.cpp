#include "CUDA.hpp"
#include "Tensor.hpp"
#include <cstring>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <ostream>
#include <chrono>

Tensor<int> originTensor(const std::vector<int>& shape, const std::string& device_type) {
    Tensor<int> tensor(shape, device_type);
    
    int *data = new int[tensor.num_elements];

    for(auto i=0; i<tensor.num_elements; i++)
        data[i] = i;

    if (device_type == "cpu") {
        std::memcpy(tensor.device->getDataPtr(), data,  tensor.num_elements * sizeof(int));
    } else if (device_type == "cuda") {
        CUDA_CHECK(cudaMemcpy(tensor.device->getDataPtr(), data, tensor.num_elements * sizeof(int), cudaMemcpyHostToDevice))
    }

    delete [] data;

    return tensor;
}

void test_matmul() {
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

void test_contiguous() {
    Tensor<int> a = originTensor({3, 4}, "cpu");
    Tensor<int> b = originTensor({3, 4}, "cuda");

    std::cout << "a: " << std::endl << a << std::endl;
    std::cout << "b: " << std::endl << b << std::endl;

    a = a.transpose(0, 1);
    b = b.transpose(0, 1);

    std::cout << "a: " << std::endl << a << std::endl;
    for (int i = 0; i < a.num_elements; i++) {
        std::cout << a.device->getDataLinear(i) << " ";
    }
    std::cout << std::endl;

    std::cout << "b: " << std::endl << b << std::endl;
    for (int i = 0; i < b.num_elements; i++) {
        std::cout << b.device->getDataLinear(i) << " ";
    }
    std::cout << std::endl;

    // seems have bug...
    a = a.contiguous();
    b = b.contiguous();

    std::cout << "a: " << std::endl << a << std::endl;
    for (int i = 0; i < a.num_elements; i++) {
        std::cout << a.device->getDataLinear(i) << " ";
    }
    std::cout << std::endl;

    std::cout << "b: " << std::endl << b << std::endl;
    for (int i = 0; i < b.num_elements; i++) {
        std::cout << b.device->getDataLinear(i) << " ";
    }
    std::cout << std::endl;

}

void test_unary_op() {
    auto a = originTensor(std::vector<int>{2, 3}, "cpu");
    auto b = originTensor(std::vector<int>{2, 3}, "cuda");

    std::cout << "a: " << std::endl << a << std::endl;
    std::cout << "b: " << std::endl << b << std::endl;

    a = -a;
    std::cout << "a: " << std::endl << a << std::endl;

    b = -b;
    std::cout << "b: " << std::endl << b << std::endl;
}

void test_reduce_op() {
    auto a = originTensor(std::vector<int>{2, 3}, "cpu");

    std::cout << "a: " << std::endl << a << std::endl;

    auto b = a.max(0, true);
    std::cout << "b: " << std::endl << b << std::endl;

    auto c = a.max(1, true);
    std::cout << "c: " << std::endl << c << std::endl;

    auto d = a.max({}, true);
    std::cout << "d: " << std::endl << d << std::endl;
}

int main() {
    // test_contiguous();
    // test_unary_op();
    test_reduce_op();
}
