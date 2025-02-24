// this file provides single precision matrix multiplication kernels(Sgemm).
#include "device/cuda/CUDA.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// may have error...
void sgemm_cublas(const float* lhs, const float* rhs, float* result, size_t M, size_t N, size_t K) {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        float alpha = 1.0f;
        float beta = 0.0f;

        CUBLAS_CHECK(cublasSgemm(
            handle, 
            CUBLAS_OP_T, CUBLAS_OP_N, 
            N, M, K, 
            &alpha, 
            rhs, N, 
            lhs, K, 
            &beta, 
            result, N));

        CUBLAS_CHECK(cublasDestroy(handle));
}
