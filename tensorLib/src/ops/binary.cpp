// 
// #include "Tensor.hpp"
// 
// template class Tensor<int>;
// template class Tensor<float>;
// template class Tensor<half>;
// template class Tensor<int8_t>;
// 
// ////////////////////////////////////////////////////// binary operations ///////////////////////////////////////////////////////////////////////////////
// /**
//  * used for implicit broadcasting 
//  * implicitly broadcasting before operation, for example:
//  * a(5, 1, 3) + b(4, 3) -> a(5, 1, 3) + b(1, 4, 3) -> new_shape(5, 4, 3)
//  */
// template <typename dtype>
// std::vector<int> Tensor<dtype>::get_broadcast_shape(std::vector<int>& shape_a, std::vector<int>& shape_b) const {
//     if (shape_a == shape_b) return shape_a;
// 
//     auto a = shape_a;
//     auto b = shape_b;
// 
//     int dims_a = shape_a.size();
//     int dims_b = shape_b.size();
//     int max_dims = std::max(dims_a, dims_b);
// 
//     if (dims_a > dims_b) {
//         for (int i=dims_b; i < dims_a; i++) {
//             b.insert(b.begin(), 1);
//         }
//     } else {
//         for (int i=dims_a; i < dims_b; i++) {
//             a.insert(a.begin(), 1);
//         }
//     }
// 
//     // now a.size() == b.size()
//     std::vector<int> new_shape;
//     for (int i=0; i <max_dims; i++) {
//         if (a[i] == b[i]) {
//             new_shape.push_back(a[i]);
//         } else if (a[i] == 1) {
//             new_shape.push_back(b[i]);
//         } else if (b[i] == 1) {
//             new_shape.push_back(a[i]);
//         } else {
//             throw std::invalid_argument("The shape cannot be broadcasted.");
//         }
//     }
// 
//     return new_shape;
// }
// 
// /**
//  * maybe should support implicit broadcasting 
//  * @tparam dtype 
//  */
// template <typename dtype>
// template <void (Device<dtype>::*func)(dtype*, dtype*, size_t) const>
// Tensor<dtype> Tensor<dtype>::applyBinaryOperation(const Tensor<dtype>& other) const {
//     if (this->device_type != other.device_type) {
//         throw std::invalid_argument("The device type of the two tensors must be the same.");
//     }
// 
//     Tensor<dtype> a = *this, b = other;
//     
//     // implicit broadcasting
//     if (this->shape() != other.shape()) {
//         std::vector<int> shape_a = this->shape();
//         std::vector<int> shape_b = other.shape();
//         auto new_shape = get_broadcast_shape(shape_a, shape_b);
//         a = a.broadcast_to(new_shape); 
//         b = b.broadcast_to(new_shape);
//     }
// 
//     // maybe we can do not call contiguous() to make it faster...
//     /**
//         import torch
//         # Create a base tensor and a non-contiguous tensor
//         a = torch.rand(3, 1)   # Shape (3, 1)
//         b = torch.rand(3, 3).permute(1, 0)  # Shape (3, 3) but non-contiguous
// 
//         # Perform an operation that requires broadcasting
//         result = a + b  # Broadcasting will occur here
// 
//         print(result)
//         print(result.is_contiguous())  # The result might be non-contiguous if any input was non-contiguous
//      */
// 
//     a = a.contiguous();
//     b = b.contiguous();
// 
//     Tensor<dtype> result(this->shape(), this->device_type);
//     // (this->device.get()->*func)(result.device->getDataPtr(), other.device->getDataPtr(), result.num_elements); // error!!
//     (a.device.get()->*func)(result.device->getDataPtr(), b.device->getDataPtr(), result.num_elements);
//     return result;
// }
// 
// template <typename dtype> Tensor<dtype> Tensor<dtype>::operator+(const Tensor<dtype>& other) const { return applyBinaryOperation<&Device<dtype>::add>(other); }
// template <typename dtype> Tensor<dtype> Tensor<dtype>::operator-(const Tensor<dtype>& other) const { return applyBinaryOperation<&Device<dtype>::sub>(other); }
// template <typename dtype> Tensor<dtype> Tensor<dtype>::operator*(const Tensor<dtype>& other) const { return applyBinaryOperation<&Device<dtype>::mul>(other); }
// template <typename dtype> Tensor<dtype> Tensor<dtype>::operator/(const Tensor<dtype>& other) const { return applyBinaryOperation<&Device<dtype>::div>(other); }
// 
// /**
//  * if this is a broadcasted tensor, need to use contiguous() firtst, or the num_elements is not the actual elem size of the tensor' device data_.
//  * @tparam dtype 
//  */
// template <typename dtype>
// template <void (Device<dtype>::*func)(dtype*, dtype, size_t) const>
// Tensor<dtype> Tensor<dtype>::applyBinaryScalarOperation(dtype scalar) const {
//     // Tensor<dtype> result = this->contiguous();
//     Tensor<dtype> result(this->shape(), this->device_type);
//     Tensor<dtype> this_contiguous = this->contiguous();//if do not contiguous broadcast may get error, for it does not have num_elements elems actually.
//     (this_contiguous.device.get()->*func)(result.device->getDataPtr(), scalar, result.num_elements);
//     return result;
// }
// 
// template <typename dtype> Tensor<dtype> Tensor<dtype>::operator+(dtype scalar) const { return applyBinaryScalarOperation<&Device<dtype>::add>(scalar); }
// template <typename dtype> Tensor<dtype> Tensor<dtype>::operator-(dtype scalar) const { return applyBinaryScalarOperation<&Device<dtype>::sub>(scalar); }
// template <typename dtype> Tensor<dtype> Tensor<dtype>::operator*(dtype scalar) const { return applyBinaryScalarOperation<&Device<dtype>::mul>(scalar); }
// template <typename dtype> Tensor<dtype> Tensor<dtype>::operator/(dtype scalar) const { return applyBinaryScalarOperation<&Device<dtype>::div>(scalar); }
// template <typename dtype> Tensor<dtype> Tensor<dtype>::pow(dtype scalar) const { return applyBinaryScalarOperation<&Device<dtype>::pow>(scalar); }
// 
