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

    // the memory pointed by result will be modified, can not mark it as const
    virtual void matmul(dtype* lhs, dtype* rhs, dtype* result,
        std::vector<int>& lhs_stride, 
        std::vector<int>& rhs_stride, 
        size_t lhs_offset,
        size_t rhs_offset,
        std::vector<int>& result_shape,
        size_t result_elements,
        size_t K) = 0; // pure virtual
    
    virtual dtype* getData() = 0;  // Pure virtual getter for data_

// private:
};

