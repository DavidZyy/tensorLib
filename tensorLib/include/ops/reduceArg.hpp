#pragma once
#include "Tensor.hpp"

namespace ops {

template <typename dtype, void (Device<dtype>::*func)(int*, size_t, size_t) const>
struct ReduceArg {
    static Tensor<dtype> call(Tensor<dtype> const& x, int axis);
    
};

}
