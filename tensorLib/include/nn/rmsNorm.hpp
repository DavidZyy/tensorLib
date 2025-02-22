#pragma once
#include "nn/modules.hpp"

namespace nn {

template <typename dtype>
class RMSNorm : public nn::Module<dtype> {
public:
    RMSNorm() = default;
    RMSNorm(int dim, float eps = 1e-5, std::string device_type = "cpu");

    Tensor<dtype> _norm(Tensor<dtype> x) const;

    // Tensor<dtype> forward(const Tensor<dtype>& x, std::optional<Tensor<dtype>> result_opt = std::nullopt) const override;
    Tensor<dtype> forward(const Tensor<dtype>& x, std::optional<Tensor<dtype>> result_opt = std::nullopt) const;
    Tensor<dtype> forward_plain(const Tensor<dtype>& x, std::optional<Tensor<dtype>> result_opt = std::nullopt) const; // support cpu and cuda
    Tensor<dtype> forward_fused_cuda(const Tensor<dtype>& x, std::optional<Tensor<dtype>> result_opt = std::nullopt) const; // only support cuda
// private:
    float eps;
    int dim;
    Tensor<dtype> weight;
};

} // namespace nn
