#include "Device.hpp"
#include "CPU.hpp"
#include "omp.h"
#include <cstddef>


/**
 * matmul operation on CPU
 * @tparam dtype 
 */
template<typename dtype>
void CPU<dtype>::matmul(dtype* lhs, dtype* rhs, dtype* result, 
        std::vector<int>& lhs_stride, 
        std::vector<int>& rhs_stride, 
        size_t lhs_offset,
        size_t rhs_offset,
        std::vector<int>& result_shape, 
        size_t result_elements,
        size_t K
        ) {

    size_t ndim = result_shape.size();

    #pragma omp parallel for
    for (size_t idx = 0; idx < result_elements; ++idx) {

        size_t linear_index = idx;
        size_t Aoff = lhs_offset, Boff = rhs_offset;
        int row, col;

        for (int i = ndim - 1; i >= 0; --i) {
            int cur_dim_id = linear_index % result_shape[i];
            linear_index /= result_shape[i];

            if (i != ndim - 1)
                Aoff += cur_dim_id * lhs_stride[i];
            if (i != ndim - 2)
                Boff += cur_dim_id * rhs_stride[i];
        }

        dtype sum = 0;
        int t1 = lhs_stride[ndim - 1], t2 = rhs_stride[ndim - 2];

        for (int k = 0; k < K; ++k) {
            sum += lhs[Aoff + k * t1] * rhs[Boff + k * t2];
        }

        result[idx] = sum;
    }

}
