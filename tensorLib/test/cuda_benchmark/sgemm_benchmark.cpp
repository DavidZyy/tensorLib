// reference: https://github.com/nicolaswilde/cuda-sgemm

#include "device/Device.hpp"
#include "device/CPU.hpp"
#include "device/CUDA.hpp"
#include <ctime>
#include <iostream>

#define M 1024
#define N 1024
#define K 1024

CPU<float> cpu;
CUDA<float> cuda;

// #define OFFSET(row, col, ld) ((row) * (ld) + (col))

/**
 * test the max error with cpu and cuda
 */
// template <>
float testMaxError() {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    CUDA_CHECK(cudaMalloc(&d_a, size_a));
    CUDA_CHECK(cudaMalloc(&d_b, size_b));
    CUDA_CHECK(cudaMalloc(&d_c, size_c));
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);

    cpu.matmul2d(h_a, h_b, h_c, M, N, K);
    // cuda.matmul2d_Cublas(d_a, d_b, d_c, M, N, K);
    cuda.matmul2d(d_a, d_b, d_c, M, N, K);

    CUDA_CHECK(cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost));

    float max_error = 0;
    for (int i = 0; i < M * N; i++) {
        float this_error = std::abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = std::max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_d_c);

    return max_error;
}


int main() {
    float max_error = testMaxError();
    std::cout << "max error: " << max_error << std::endl;
    return 0;
}