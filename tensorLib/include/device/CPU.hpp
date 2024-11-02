#pragma once

#include "Device.hpp"
#include <cstddef>
#include <vector>

template <typename dtype>
class CPU : public Device<dtype> {
public:
    // Default constructor
    CPU() : data_(nullptr) {}
    CPU(dtype *ptr) {data_ = ptr;}
    CPU(size_t num_elements) { data_ = new dtype[num_elements]; }
    ~CPU() override { delete[] data_; }

    void matmul(const dtype* lhs, const dtype* rhs, dtype* result, 
        const std::vector<int>& lhs_stride, 
        const std::vector<int>& rhs_stride, 
        size_t lhs_offset,
        size_t rhs_offset,
        const std::vector<int>& result_shape,
        size_t result_elements,
        size_t K) override;

    dtype* getDataPtr() override { return data_; }
    void full (size_t num_elements, dtype fill_value) override;
    dtype getDataLinear(size_t liner_index) const override;

// private:
    dtype *data_;
};
