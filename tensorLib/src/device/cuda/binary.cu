#include "device/cuda/CUDA.cuh"
#include "device/cuda/binary.cuh"


template class CUDA<int8_t>;
template class CUDA<half>;
template class CUDA<float>;
template class CUDA<int>;

template <typename dtype> void CUDA<dtype>::add(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<addFunc<dtype>>(result, other, num_elements);}
template <typename dtype> void CUDA<dtype>::sub(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<subFunc<dtype>>(result, other, num_elements);}
template <typename dtype> void CUDA<dtype>::mul(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<mulFunc<dtype>>(result, other, num_elements);}
template <typename dtype> void CUDA<dtype>::div(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<divFunc<dtype>>(result, other, num_elements);}

template <typename dtype> void CUDA<dtype>::add(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<addFunc<dtype>>(result, value, num_elements);}
template <typename dtype> void CUDA<dtype>::sub(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<subFunc<dtype>>(result, value, num_elements);}
template <typename dtype> void CUDA<dtype>::mul(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<mulFunc<dtype>>(result, value, num_elements);}
template <typename dtype> void CUDA<dtype>::div(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<divFunc<dtype>>(result, value, num_elements);}
template <typename dtype> void CUDA<dtype>::pow(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<powFunc<dtype>>(result, value, num_elements);}

