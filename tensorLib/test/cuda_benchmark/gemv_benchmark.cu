#include "Tensor.hpp"
#include "device/CUDA.hpp"

#define M 1
// #define N (16)
// #define K (16)
// #define N (4096 * 1)
// #define K (4096 * 1)
#define N (11008)
#define K (11008)

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
        if (fabs(a - b) >= 1e-1) {
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

template<typename dtype> void gemv_v0(const dtype* A, const dtype* B, dtype* C, size_t n, size_t k);
template<typename dtype> void gemv_v3(const dtype* A, const dtype* B, dtype* C, size_t n, size_t k);
template<typename dtype> void gemv_v4(const dtype* A, const dtype* B, dtype* C, size_t n, size_t k);
template<typename dtype> void gemv_v5(const dtype* A, const dtype* B, dtype* C, size_t n, size_t k);
template<typename dtype> void gemv_v6(const dtype* A, const dtype* B, dtype* C, size_t n, size_t k);
template<typename dtype> void gemv_v7(const dtype* A, const dtype* B, dtype* C, size_t n, size_t k);
template<typename dtype> void gemv_v8(const dtype* A, const dtype* B, dtype* C, size_t n, size_t k);
// template<typename dtype> void gemv_v9(const dtype* A, const dtype* B, dtype* C, size_t n, size_t k);
template<typename dtype> void gemv_cublasSgemv(const dtype* A, const dtype* B, dtype* C, size_t n, size_t k);

/************************************************************************************************************************************************************/
template<typename dtype, typename GemvFunc>
float timer(GemvFunc gemv, Tensor<dtype> A, Tensor<dtype> B, Tensor<dtype> C, int repeat) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++)
        gemv(A.device->getDataPtr(), B.device->getDataPtr(), C.device->getDataPtr(), N, K);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float msec, sec;
    CUDA_CHECK(cudaEventElapsedTime(&msec, start, end));
    sec = msec / 1000.0;
    return sec;
}

template<typename dtype, typename GemvFunc>
void profile(GemvFunc gemv, Tensor<dtype> A, Tensor<dtype> B, Tensor<dtype> C, Tensor<dtype> C_r, int repeat, std::string name) {
    std::cout << "------------------" << name << "------------------" << std::endl;

    float total_sec = timer(gemv, A, B, C, repeat);

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

int main() {
    Tensor<half> A = randn<half>({M, K}, "cuda");
    Tensor<half> B = randn<half>({K, N}, "cuda");
    // Tensor<half> A = full<half>({M, K}, 1, "cuda");
    // Tensor<half> B = full<half>({K, N}, 1, "cuda");
    Tensor<half> C = A.matmul(B.transpose(0, 1)); // B is col major, a little mess ... We can think that C is correctness

    Tensor<half> C0 = full<half>({M, N}, 0, "cuda");
    Tensor<half> C3 = full<half>({M, N}, 0, "cuda");
    Tensor<half> C4 = full<half>({M, N}, 0, "cuda");
    Tensor<half> C6 = full<half>({M, N}, 0, "cuda");
    Tensor<half> C7 = full<half>({M, N}, 0, "cuda");
    Tensor<half> C8 = full<half>({M, N}, 0, "cuda");
    // Tensor<half> C9 = full<half>({M, N}, 0, "cuda");

    int repeat = 100;

    profile(gemv_v0<half>, A, B, C0, C, repeat, "gemv_v0");
    profile(gemv_v3<half>, A, B, C3, C, repeat, "gemv_v3");
    profile(gemv_v4<half>, A, B, C4, C, repeat, "gemv_v4");
    profile(gemv_v6<half>, A, B, C6, C, repeat, "gemv_v6");
    // profile(gemv_v7<half>, A, B, C7, C, repeat, "gemv_v7");
    profile(gemv_v8<half>, A, B, C8, C, repeat, "gemv_v8");
    // profile(gemv_v9<half>, A, B, C8, C, repeat, "gemv_v9");

}

// int main() {
//     Tensor<float> A = randn<float>({M, K}, "cuda");
//     Tensor<float> B = randn<float>({K, N}, "cuda");
//     // Tensor<float> A = full<half>({M, K}, 1, "cuda");
//     // Tensor<float> B = full<half>({K, N}, 1, "cuda");
//     Tensor<float> C = A.matmul(B.transpose(0, 1)); // B is col major, a little mess ... We can think that C is correctness
// 
//     Tensor<float> C0 = full<float>({M, N}, 0, "cuda");
//     Tensor<float> C3 = full<float>({M, N}, 0, "cuda");
//     Tensor<float> C4 = full<float>({M, N}, 0, "cuda");
//     Tensor<float> C5 = full<float>({M, N}, 0, "cuda");
//     Tensor<float> C_cublasSgemv = full<float>({M, N}, 0, "cuda");
// 
//     int repeat = 50;
// 
//     profile(gemv_v0<float>, A, B, C0, C, repeat, "gemv_v0");
//     profile(gemv_v3<float>, A, B, C3, C, repeat, "gemv_v3");
//     profile(gemv_v4<float>, A, B, C4, C, repeat, "gemv_v4");
//     profile(gemv_v5<float>, A, B, C5, C, repeat, "gemv_v5");
//     profile(gemv_cublasSgemv<float>, A, B, C_cublasSgemv, C, repeat, "gemv_cublasSgemv");
// }
