#pragma once
#include "nn/modules.hpp"

namespace nn {

template <typename dtype> 
class Embedding : public Module<dtype> {
public:
  Embedding() = default;
  Embedding(int num_embeddings, int embedding_dim, std::string device_type);
  ~Embedding() = default;

  // Tensor<dtype> forward(const Tensor<int> &input) const;
  Tensor<dtype> forward(const Tensor<int> &input, std::optional<Tensor<dtype>> result_opt = std::nullopt) const;

  // private:
  // protected:
  int num_embeddings;
  int embedding_dim;
  Tensor<dtype> weight;
};

} // namespace nn
