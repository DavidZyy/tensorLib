#pragma once
#include <cstddef>
#include <memory>
#include <vector>

template <typename dtype>
class Device {
public:
    // Device() = default;
    // Device(size_t num_elements);
    virtual ~Device() = default;

    virtual void matmul(const dtype* lhs, const dtype* rhs, dtype* result, 
        const std::vector<int>& lhs_stride, 
        const std::vector<int>& rhs_stride, 
        size_t lhs_offset,
        size_t rhs_offset,
        const std::vector<int>& result_shape,
        size_t result_elements,
        size_t K) = 0;

    virtual dtype* getDataPtr() = 0;  // Pure virtual getter for data_
    virtual void full (size_t num_elements, dtype fill_value) = 0;
    // get data by linear index
    virtual dtype getDataLinear(size_t liner_index) const = 0;

// private:
};

