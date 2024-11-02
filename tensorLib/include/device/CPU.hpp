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
    void contiguous(
        dtype* result, 
        const std::vector<int>& shape,
        const std::vector<int>& stride, 
        size_t offset,
        size_t num_elements) override;

// private:
    dtype *data_;
    inline size_t convertIdx(size_t linear_index, const std::vector<int>& shape, const std::vector<int>& stride, size_t offset) const {
        size_t linear_index_new = 0;

        for (int i = shape.size() - 1; i >= 0; --i) {
            int cur_dim_id = linear_index % shape[i];
            linear_index /= shape[i];
            linear_index_new += cur_dim_id * stride[i];
        }

        return linear_index_new + offset;
    }

};
