#pragma once

#include "Device.hpp"
#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename dtype>
class CUDA : public Device<dtype> {
public:
    CUDA() : data_(nullptr) {}
    CUDA(size_t num_elements);
    ~CUDA() override;

    // void matmul(const dtype* lhs, const dtype* rhs, dtype* result, 
    //     const std::vector<int>& lhs_stride, 
    //     const std::vector<int>& rhs_stride, 
    //     size_t lhs_offset,
    //     size_t rhs_offset,
    //     const std::vector<int>& result_shape,
    //     size_t result_elements,
    //     size_t K) override;
    
    void matmul(dtype* lhs, dtype* rhs, dtype* result, 
        std::vector<int>& lhs_stride, 
        std::vector<int>& rhs_stride, 
        size_t lhs_offset,
        size_t rhs_offset,
        std::vector<int>& result_shape,
        size_t result_elements,
        size_t K) override;

    dtype* getData() override { return data_; }


// private:
    // std::shared_ptr<dtype[]> data_;
    dtype *data_;
    // size_t num_elements;
};
