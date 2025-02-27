// #include "ops/reduce.hpp"
// #include "device/Device.hpp"
// 
// // seems use struct ops is too complex and make no sense, we can not use this ...
// 
// namespace ops {
// template struct reduce<int8_t, int8_t, &Device<int8_t>::max>;
// template struct reduce<half, half, &Device<half>::max>;
// template struct reduce<float, float, &Device<float>::max>;
// template struct reduce<int, int, &Device<int>::max>;
// 
// template struct reduce<int8_t, int8_t, &Device<int8_t>::min>;
// template struct reduce<half, half, &Device<half>::min>;
// template struct reduce<float, float, &Device<float>::min>;
// template struct reduce<int, int, &Device<int>::min>;
// 
// template struct reduce<int8_t, int8_t, &Device<int8_t>::sum>;
// template struct reduce<half, half, &Device<half>::sum>;
// template struct reduce<float, float, &Device<float>::sum>;
// template struct reduce<int, int, &Device<int>::sum>;
// 
// template struct reduce<int8_t, int8_t, &Device<int8_t>::mean>;
// template struct reduce<half, half, &Device<half>::mean>;
// template struct reduce<float, float, &Device<float>::mean>;
// template struct reduce<int, int, &Device<int>::mean>;
// 
// 
// template struct reduce<half, int, &Device<half>::argmax>;
// template struct reduce<float, int, &Device<float>::argmax>;
// 
// template struct reduce<half, int, &Device<half>::argmin>;
// template struct reduce<float, int, &Device<float>::argmin>;
// 
// 
// 
// 
// }


#include "Tensor.hpp"

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<half>;
template class Tensor<int8_t>;


template <typename dtype>
int Tensor<dtype>::handle_axis(int axis) const {
    int dims = static_cast<int>(this->shape().size()); // size is unsigned, so use int
    if (axis >= dims) {
        throw std::invalid_argument("The axis must be less than the shape size.");
    } else if (axis < -dims) {
        throw std::invalid_argument("The axis must be greater than or equal to -shape size.");
    }

    if (axis < 0) {
        axis += this->shape().size();
    }
    return axis;
}

/**
 * permute the axis to the last dimension first, then return a view of the tensor which is contiguous.
 * @tparam dtype 
 */
template<typename dtype>
Tensor<dtype> Tensor<dtype>::get_reduce_view(int axis) const {
    std::vector<int> new_axes;
    for (int i=0; i < this->shape().size(); i++) {
        if (i != axis) {
            new_axes.push_back(i);
        }
    }
    new_axes.push_back(axis);

    auto view = this->permute(new_axes);
    view = view.contiguous();
    return view;
}

template<typename dtype>
std::vector<int> Tensor<dtype>::get_reduce_shape(int axis, bool keepdims) const {
    std::vector<int> new_shape = this->shape();
    if (keepdims) {
        new_shape[axis] = 1;
    } else {
        new_shape.erase(new_shape.begin() + axis);
    }
    
    return new_shape;
}

template<typename dtype>
template<typename Rtype, void (Device<dtype>::*func)(Rtype*, size_t, size_t) const>
Tensor<Rtype> Tensor<dtype>::reduceOperation(std::optional<int> axis, bool keepdims) const {
    Tensor<dtype> view;
    std::vector<int> new_shape;
    int reduce_size;

    if (axis.has_value()) {
        int axis_v = handle_axis(axis.value());
        view = get_reduce_view(axis_v); // Permute to move the reduction axis to the last dimension
        new_shape = get_reduce_shape(axis_v, keepdims);
        reduce_size = this->shape()[axis_v];
    } else {
        // Reduce all dimensions
        view = *this;
        if (keepdims)
            new_shape = std::vector<int>(this->shape().size(), 1);
        reduce_size = this->num_elements;
    }

    Tensor<Rtype> result(new_shape, this->device_type);

    // Call the device function with the appropriate pointer, reduce size, and total elements
    (view.device.get()->*func)(result.device->getDataPtr(), reduce_size, this->num_elements);

    return result;
}

template<typename dtype> 
Tensor<dtype> Tensor<dtype>::max(std::optional<int> axis, bool keepdims) const { 
    return reduceOperation<dtype, &Device<dtype>::max>(axis, keepdims);
}

template<typename dtype> 
Tensor<dtype> Tensor<dtype>::min(std::optional<int> axis, bool keepdims) const { 
    return reduceOperation<dtype, &Device<dtype>::min>(axis, keepdims);
}

template<typename dtype> 
Tensor<dtype> Tensor<dtype>::sum(std::optional<int> axis, bool keepdims) const { 
    return reduceOperation<dtype, &Device<dtype>::sum>(axis, keepdims);
}

template<typename dtype> 
Tensor<dtype> Tensor<dtype>::mean(std::optional<int> axis, bool keepdims) const { 
    return reduceOperation<dtype, &Device<dtype>::mean>(axis, keepdims);
}

template<typename dtype> 
Tensor<int> Tensor<dtype>::argmax(std::optional<int> axis, bool keepdims) const { 
    return reduceOperation<int, &Device<dtype>::argmax>(axis, keepdims);
}

template<typename dtype> 
Tensor<int> Tensor<dtype>::argmin(std::optional<int> axis, bool keepdims) const { 
    return reduceOperation<int, &Device<dtype>::argmin>(axis, keepdims);
}
