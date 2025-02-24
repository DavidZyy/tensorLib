#include "Tensor.hpp"
#include "device/cuda/CUDA.cuh"
#include "test.hpp"

#define M (4096)
#define N (4096)
#define K (4096)

/************************************************************************************************************************************************************/
template<typename dtype, typename GemmFunc>
float timer(GemmFunc gemm, Tensor<dtype> A, Tensor<dtype> B, Tensor<dtype> C, int repeat) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++)
        gemm(A.device->getDataPtr(), B.device->getDataPtr(), C.device->getDataPtr(), M, N, K);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float msec, sec;
    CUDA_CHECK(cudaEventElapsedTime(&msec, start, end));
    sec = msec / 1000.0;
    return sec;
}

template<typename dtype, typename GemmFunc>
void profile(GemmFunc gemm, Tensor<dtype> A, Tensor<dtype> B, Tensor<dtype> C, Tensor<dtype> C_r, int repeat, std::string name) {
    std::cout << "------------------" << name << "------------------" << std::endl;

    float total_sec = timer(gemm, A, B, C, repeat);

    double avg_sec = total_sec / repeat;
    double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

    std::cout << "average time: " << avg_sec << "s" << std::endl;
    std::cout << "average Gflops: " << avg_Gflops << std::endl;
    
    check_equal_and_max_diff(C, C_r);
}

/************************************************************************************************************************************************************/

// dtype
template<typename dtype> void gemm_v0(const dtype* lhs, const dtype* rhs, dtype* result, size_t m, size_t n, size_t k);

// sgemm
void sgemm_cublas(const float* lhs, const float* rhs, float* result, size_t m, size_t n, size_t k);

// hgemm
void hgemm_cublas(const half* lhs, const half* rhs, half* result, size_t m, size_t n, size_t k);
void hgemm_v0(const half *A, const half *B, half *C, size_t m, size_t n, size_t k);

/************************************************************************************************************************************************************/

int main() {
    Tensor<half> A = randn<half>({M, K}, "cuda");
    Tensor<half> B = randn<half>({K, N}, "cuda");
    Tensor<half> C = A.matmul(B.transpose(0, 1)); // B is col major, a little mess ... We can think that C is correctness

    Tensor<half> C0 = full<half>({M, N}, 1, "cuda");
    Tensor<half> C1 = full<half>({M, N}, 1, "cuda");
    Tensor<half> C2 = full<half>({M, N}, 1, "cuda");

    int repeat = 20;
    profile(gemm_v0<half>, A, B, C0, C, repeat, "gemm_v0");
    profile(hgemm_cublas, A, B, C1, C, repeat, "hgemm_cublas");
    profile(hgemm_v0, A, B, C2, C, repeat, "hgemm_v0");

    return 0;
}

// int main() {
//     Tensor<float> A = randn<float>({M, K}, "cuda");
//     Tensor<float> B = randn<float>({K, N}, "cuda");
//     Tensor<float> C = A.matmul(B.transpose(0, 1)); // B is col major, a little mess ... We can think that C is correctness
// 
//     Tensor<float> C0 = full<float>({M, N}, 0, "cuda");
//     Tensor<float> C1 = full<float>({M, N}, 0, "cuda");
// 
//     int repeat = 20;
//     profile(gemm_v0<float>, A, B, C0, C, repeat, "gemm_v0");
//     profile(sgemm_cublas, A, B, C1, C, repeat, "sgemm_cublas");
// 
//     return 0;
// }
