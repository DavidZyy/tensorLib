// this file provides half precision matrix multiplication kernels using tensor core.
#include "device/CUDA.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// may have error...
void hgemm_cublas(const half* lhs, const half* rhs, half* result, size_t M, size_t N, size_t K) {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        half alpha = 1;
        half beta = 0;

        CUBLAS_CHECK(cublasHgemm(
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
