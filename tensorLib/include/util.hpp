#pragma once

#include "cuda_runtime_api.h"

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    // return (a % b != 0) ? (a / b + 1) : (a / b);
    return (a + b - 1) / b;
}
