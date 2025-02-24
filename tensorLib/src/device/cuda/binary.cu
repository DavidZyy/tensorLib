#include "device/cuda/CUDA.cuh"
#include "device/cuda/binary.cuh"


template class CUDA<int8_t>;
template class CUDA<half>;
template class CUDA<float>;
template class CUDA<int>;

template <typename dtype> void CUDA<dtype>::add(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<dtype, addFunc<dtype>>(result, this->data_, other, num_elements);}
template <typename dtype> void CUDA<dtype>::sub(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<dtype, subFunc<dtype>>(result, this->data_, other, num_elements);}
template <typename dtype> void CUDA<dtype>::mul(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<dtype, mulFunc<dtype>>(result, this->data_, other, num_elements);}
template <typename dtype> void CUDA<dtype>::div(dtype* result, dtype* other, size_t num_elements) const {applyBinaryOperation<dtype, divFunc<dtype>>(result, this->data_, other, num_elements);}

template <typename dtype> void CUDA<dtype>::add(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<dtype, addFunc<dtype>>(result, this->data_, value, num_elements);}
template <typename dtype> void CUDA<dtype>::sub(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<dtype, subFunc<dtype>>(result, this->data_, value, num_elements);}
template <typename dtype> void CUDA<dtype>::mul(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<dtype, mulFunc<dtype>>(result, this->data_, value, num_elements);}
template <typename dtype> void CUDA<dtype>::div(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<dtype, divFunc<dtype>>(result, this->data_, value, num_elements);}
template <typename dtype> void CUDA<dtype>::pow(dtype* result, dtype value, size_t num_elements) const {applyBinaryScalarOperation<dtype, powFunc<dtype>>(result, this->data_, value, num_elements);}
