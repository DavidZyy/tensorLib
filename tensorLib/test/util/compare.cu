#include "device/CUDA.hpp"
#include "Tensor.hpp"

/************************************************************************************************************************************************************/

__device__ static float atomicMax(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

template <typename T>
__global__ void compare_max_kernel(const T* a_data, const T* b_data, size_t num_elements, float* max_diff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id = threadIdx.x;

    // Allocate shared memory for reduction within a block
    __shared__ float shared_max[256];  // Assuming block size of 256

    // Initialize the shared memory
    if (thread_id < 256) {
        shared_max[thread_id] = 0.0f;
    }
    __syncthreads();

    // Perform computation for each element
    if (idx < num_elements) {
        // Convert half to float for comparison
        float a = __half2float(a_data[idx]);
        float b = __half2float(b_data[idx]);

        // Calculate the absolute difference
        float diff = fabs(a - b);

        // Store the difference in the shared memory
        shared_max[thread_id] = diff;
    }
    __syncthreads();

    // Perform block-wide reduction to find the max difference within this block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_id < s) {
            shared_max[thread_id] = fmaxf(shared_max[thread_id], shared_max[thread_id + s]);
        }
        __syncthreads();
    }

    // The thread with thread_id == 0 will write the result to global memory
    if (thread_id == 0) {
        atomicMax(max_diff, shared_max[0]);
    }
}

template <typename T>
float get_max_abs_difference(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.num_elements != b.num_elements) {
        std::cerr << "Tensors have different sizes!" << std::endl;
        return -1.0f; // Return an error value if tensors are not equal in size
    }

    // Allocate memory for max_diff on the device
    float* d_max_diff;
    float h_max_diff = 0.0f; // Initialize to 0 (no difference yet)
    cudaMalloc(&d_max_diff, sizeof(float));
    cudaMemcpy(d_max_diff, &h_max_diff, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to compare the tensors and track the max absolute difference
    int block_size = 256; // Choose an appropriate block size (must be a power of 2)
    int grid_size = (a.num_elements + block_size - 1) / block_size;
    compare_max_kernel<<<grid_size, block_size>>>(a.device->getDataPtr(), b.device->getDataPtr(), a.num_elements, d_max_diff);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the max difference back to the host
    cudaMemcpy(&h_max_diff, d_max_diff, sizeof(float), cudaMemcpyDeviceToHost);

    // Free the device memory for max_diff
    cudaFree(d_max_diff);

    return h_max_diff; // Return the maximum absolute difference
}

// instantiation
template float get_max_abs_difference<half>(const Tensor<half>& a, const Tensor<half>& b);
template float get_max_abs_difference<float>(const Tensor<float>& a, const Tensor<float>& b);
/************************************************************************************************************************************************************/

// Helper CUDA kernel to compare tensors element-wise
template <typename T>
__global__ void compare_kernel(const T* a_data, const T* b_data, size_t num_elements, bool* result_flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Convert half to float for comparison
        float a = __half2float(a_data[idx]);
        float b = __half2float(b_data[idx]);

        // Compare the elements, and if they are different, set the flag
        if (fabs(a - b) >= 1e-1) {
            // printf("idx: %d, a: %f, b: %f\n", idx, a, b);
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
    // CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result flag back to the host
    cudaMemcpy(&h_result_flag, d_result_flag, sizeof(bool), cudaMemcpyDeviceToHost);

    // Free the device memory for the flag
    cudaFree(d_result_flag);

    return h_result_flag; // Return whether the tensors are equal
}

template bool check_equal<half>(const Tensor<half>& a, const Tensor<half>& b);
template bool check_equal<float>(const Tensor<float>& a, const Tensor<float>& b);
/************************************************************************************************************************************************************/

template <typename T>
void check_equal_and_max_diff(Tensor<T>& a, Tensor<T>& b) {
    if (check_equal(a, b)) {
        std::cout << "pass!" << std::endl;
    } else {
        std::cout << "failed!" << std::endl;
    }

    float max_diff = get_max_abs_difference(a, b);
    std::cout << "max_diff: " << max_diff << std::endl;
}

template void check_equal_and_max_diff<half>(Tensor<half>& a, Tensor<half>& b);
template void check_equal_and_max_diff<float>(Tensor<float>& a, Tensor<float>& b);
/************************************************************************************************************************************************************/
