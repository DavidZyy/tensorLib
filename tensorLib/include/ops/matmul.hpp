#pragma once
#include "Tensor.hpp"

namespace ops {

template <typename dtype>
struct matmul {
    // static indicates that the function belongs to the class itself rather than to any particular instance of the class. 
    // This means it can be called without creating an instance of the class.  
    static Tensor<dtype> call (const Tensor<dtype>& self, const Tensor<dtype>& other);
};

}
