#pragma once

#include "Device.hpp"
#include <cstddef>
#include <vector>

template <typename dtype>
class CPU : public Device<dtype> {
public:
    // Default constructor
    CPU() : data_(nullptr) {}
    CPU(size_t num_elements) { data_ = new dtype[num_elements]; }
    ~CPU() override { delete[] data_; }

    // void matmul(const dtype* lhs, const dtype* rhs, dtype* result, 
    //     const std::vector<int>& lhs_stride, 
    //     const std::vector<int>& rhs_stride, 
    //     size_t lhs_offset,
    //     size_t rhs_offset,
    //     const std::vector<int>& result_shape,
    //     size_t result_elements,
    //     size_t K) override;

    // device's function maily operate on data_
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
    dtype *data_;
};
