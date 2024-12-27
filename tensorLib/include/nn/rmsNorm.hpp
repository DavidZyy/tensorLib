#include "Tensor.hpp"
#include "nn/modules.hpp"

template <typename dtype>
class RMSNorm : public nn::Module<dtype> {
public:
    RMSNorm() = default;
    RMSNorm(int dim, float eps = 1e-5, std::string device_type = "cpu");

    Tensor<dtype> _norm(Tensor<dtype> x) const;

    Tensor<dtype> forward(const Tensor<dtype>& x) const override;
    Tensor<dtype> forward_fused_cuda(const Tensor<dtype>& x) const; // only support cuda
// private:
    float eps;
    int dim;
    Tensor<dtype> weight;
};

