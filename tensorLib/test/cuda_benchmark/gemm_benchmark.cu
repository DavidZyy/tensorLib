#include "Tensor.hpp"
#include "device/CUDA.hpp"

#define M (4096)
#define N (4096)
#define K (4096)

/************************************************************************************************************************************************************/

// Helper CUDA kernel to compare tensors element-wise
template <typename T>
__global__ void compare_kernel(const T* a_data, const T* b_data, size_t num_elements, bool* result_flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float a, b;
        if constexpr (std::is_same_v<T, half>) {
            // Convert half to float for comparison
            a = __half2float(a_data[idx]);
            b = __half2float(b_data[idx]);
        } else {
            a = a_data[idx];
            b = b_data[idx];
        }

        // Compare the elements, and if they are different, set the flag
        if (fabs(a - b) > 2) {
            printf("idx: %d, a: %f, b: %f\n", idx, a, b);
            *result_flag = false; // Set result flag to false if difference is found
        }
    }
}

// Function to check equality of two tensors
template <typename T>
bool check_equal(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.num_elements != b.num_elements) {
        std::cerr << "Tensors have different sizes!" << std::endl;
        return false;
    }

    // Allocate memory for a flag on the device to indicate equality
    bool* d_result_flag;
    bool h_result_flag = true; // Initialize to true (assume equality)
    cudaMalloc(&d_result_flag, sizeof(bool));
    cudaMemcpy(d_result_flag, &h_result_flag, sizeof(bool), cudaMemcpyHostToDevice);

    // Launch kernel to compare the tensors
    int block_size = 256; // Choose an appropriate block size
    int grid_size = (a.num_elements + block_size - 1) / block_size;
    compare_kernel<<<grid_size, block_size>>>(a.device->getDataPtr(), b.device->getDataPtr(), a.num_elements, d_result_flag);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result flag back to the host
    cudaMemcpy(&h_result_flag, d_result_flag, sizeof(bool), cudaMemcpyDeviceToHost);

    // Free the device memory for the flag
    cudaFree(d_result_flag);

    return h_result_flag; // Return whether the tensors are equal
}

/************************************************************************************************************************************************************/

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

    if (check_equal(C, C_r)) {
        std::cout << "pass!" << std::endl;
    } else {
        std::cout << "failed!" << std::endl;
    }
}

/************************************************************************************************************************************************************/

template<typename dtype> void gemm_v0(const dtype* lhs, const dtype* rhs, dtype* result, size_t m, size_t n, size_t k);
void sgemm_cublas(const float* lhs, const float* rhs, float* result, size_t m, size_t n, size_t k);
void hgemm_cublas(const half* lhs, const half* rhs, half* result, size_t m, size_t n, size_t k);
/************************************************************************************************************************************************************/

int main() {
    Tensor<half> A = randn<half>({M, K}, "cuda");
    Tensor<half> B = randn<half>({K, N}, "cuda");
    Tensor<half> C = A.matmul(B.transpose(0, 1)); // B is col major, a little mess ... We can think that C is correctness

    Tensor<half> C0 = full<half>({M, N}, 0, "cuda");
    Tensor<half> C1 = full<half>({M, N}, 0, "cuda");

    int repeat = 20;
    profile(gemm_v0<half>, A, B, C0, C, repeat, "gemm_v0");
    profile(hgemm_cublas, A, B, C1, C, repeat, "hgemm_cublas");

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
