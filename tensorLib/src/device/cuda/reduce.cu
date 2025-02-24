#include "device/cuda/CUDA.cuh"
#include "device/cuda/binary.cuh"
#include "device/cuda/reduce.cuh"

template class CUDA<int8_t>;
template class CUDA<half>;
template class CUDA<float>;
template class CUDA<int>;


template <typename dtype> void CUDA<dtype>::max(dtype* result, size_t reduce_size, size_t num_elements) const { 
    reduceOperation<maxFunc<dtype>>(result, reduce_size, num_elements); 
}
template <typename dtype> void CUDA<dtype>::min(dtype* result, size_t reduce_size, size_t num_elements) const { 
    reduceOperation<minFunc<dtype>>(result, reduce_size, num_elements); 
}
template <typename dtype> void CUDA<dtype>::sum(dtype* result, size_t reduce_size, size_t num_elements) const { 
    reduceOperation<sumFunc<dtype>>(result, reduce_size, num_elements); 
}
template <typename dtype> void CUDA<dtype>::mean(dtype* result, size_t reduce_size, size_t num_elements) const { 
    reduceOperation<sumFunc<dtype>>(result, reduce_size, num_elements); 
    applyBinaryScalarOperation<dtype, divFunc<dtype>>(result, result, reduce_size, num_elements/reduce_size);
}
