#pragma once
#include <cstddef>
#include <memory>
#include <vector>

template <typename dtype>
class Device {
public:
    Device() : size(0) {}
    Device(size_t num_elements) : size(num_elements) {}
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
    virtual void contiguous(
        dtype* result, 
        const std::vector<int>& shape,
        const std::vector<int>& stride, 
        size_t offset,
        size_t num_elements) = 0;
    virtual void setItemEwise(
        dtype* src,
        const std::vector<int>& shape,
        const std::vector<int>& stride,
        size_t offset,
        size_t num_elements) = 0;
    virtual void setItemScalar(
        dtype value,
        const std::vector<int>& shape,
        const std::vector<int>& stride,
        size_t offset,
        size_t num_elements) = 0;
// private:
        size_t size; // number of elements, the total bytes of data_ is: size * sizeof(dtype)
};

