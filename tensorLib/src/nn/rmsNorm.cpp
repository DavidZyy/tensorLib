#include "nn/rmsNorm.hpp"

namespace nn {

template class RMSNorm<float>;

template <typename dtype>
RMSNorm<dtype>::RMSNorm(int dim, float eps, std::string device_type) : nn::Module<dtype>(device_type), dim(dim), eps(eps) {
    // this->weight = Tensor<dtype>({dim}, device_type);
    this->weight = randn<dtype>({dim}, device_type);
}

template <typename dtype>
Tensor<dtype> RMSNorm<dtype>::_norm(Tensor<dtype> x) const {
    // std::cout << "x:" << std::endl << x << std::endl;
    auto origin_shape = x.shape();
    auto temp = x;
    // std::cout << "x:" << std::endl << x << std::endl;
    temp = temp.pow(2);
    // std::cout << "x:" << std::endl << x << std::endl;
    temp = temp.mean(-1, true);
    // std::cout << "temp:" << std::endl << temp << std::endl;
    temp = temp.broadcast_to(origin_shape);
    // std::cout << "temp:" << std::endl << temp << std::endl;
    temp = temp + this->eps;
    temp = temp.rsqrt();
    // std::cout << "x:" << std::endl << x << std::endl;
    // std::cout << "temp:" << std::endl << temp << std::endl;
    // return x * temp; 
    auto result = x * temp;
    // std::cout << result << std::endl;
    return result;
}

template <typename dtype>
Tensor<dtype> RMSNorm<dtype>::forward(const Tensor<dtype>& x) const {
    // if (this->device_type == "cuda") {
    //     return this->forward_fused_cuda(x);
    // } else {
        return this->forward_plain(x);
    // }
}

template <typename dtype>
Tensor<dtype> RMSNorm<dtype>::forward_plain(const Tensor<dtype>& x) const {
    // std::cout << x << std::endl;
    // std::cout << weight << std::endl;
    // x : (bsz, seqlen, dim)
    // weight : (dim)
    auto result1 = this->_norm(x);
    // std::cout << result1 << std::endl;

    auto new_shape = std::vector<int>(x.ndim-1, 1);
    new_shape.push_back(this->dim);

    // auto weight = this->weight.view({1, 1, this->dim});
    auto weight = this->weight.view(new_shape);

    weight = weight.broadcast_to(x.shape());
    auto result2 = result1 * weight;

    // std::cout << result2 << std::endl;
    return result2;
}

/**
 * have lager precision error compared with pytorch version than forward_plain
 * @tparam dtype 
 */
template<typename dtype>
Tensor<dtype> RMSNorm<dtype>::forward_fused_cuda(const Tensor<dtype>& x) const {
    assert(this->device_type == "cuda");
    assert(x.is_contiguous(x));
    assert(weight.is_contiguous(weight));
    assert(dim == x.shape()[x.ndim - 1]);

    Tensor<dtype> result(x.shape(), this->device_type);
    
    // use the device of x, seems not elegant
    auto cuda_device = std::dynamic_pointer_cast<CUDA<dtype>>(x.device);
    assert(cuda_device != nullptr);

    int n_tokens = x.num_elements / dim;

    cuda_device->rms_norm(result.device->getDataPtr(), x.device->getDataPtr(), weight.device->getDataPtr(), eps, dim, n_tokens);

    return result;
}

} // namespace nn