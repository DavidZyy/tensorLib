// reference: https://github.com/nicolaswilde/cuda-sgemm
#include "device/Device.hpp"
#include "device/cpu/CPU.hpp"
#include "device/cuda/CUDA.cuh"
#include <ctime>
#include <iostream>

#define M 1024 * 4
#define N 1024 * 4
#define K 1024 * 4

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

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cpu.matmul2d(h_a, h_b, h_c, M, N, K);
    cuda.matmul2d_Cublas(d_a, d_b, d_c, M, N, K);
    // cuda.matmul2d(d_a, d_b, d_c, M, N, K);

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

float testPerformance(int repeat) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size_a));
    CUDA_CHECK(cudaMalloc(&d_b, size_b));
    CUDA_CHECK(cudaMalloc(&d_c, size_c));

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++)
        // cuda.matmul2d(d_a, d_b, d_c, M, N, K);
        cuda.matmul2d_Cublas(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float msec, sec;
    CUDA_CHECK(cudaEventElapsedTime(&msec, start, end));
    sec = msec / 1000.0 / repeat;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return sec;
}

int main() {
    float max_error = testMaxError();
    std::cout << "max error: " << max_error << std::endl;

    int repeat = 10;
    float total_sec = testPerformance(repeat);
    double avg_sec = total_sec / repeat;
    double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
    std::cout << "average time: " << avg_sec << "s" << std::endl;
    std::cout << "average Gflops: " << avg_Gflops << std::endl;
    return 0;
}
