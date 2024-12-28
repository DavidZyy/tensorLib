#include "nn/embedding.hpp"

namespace  nn {

template class Embedding<float>;
template class Embedding<int>;

template <typename dtype>
Embedding<dtype>::Embedding(int num_embeddings, int embedding_dim, std::string device_type)
    : nn::Module<dtype>(device_type), num_embeddings(num_embeddings), embedding_dim(embedding_dim),
      weight(Tensor<dtype>({num_embeddings, embedding_dim}, device_type)) {}
      // weight(randn<dtype>({num_embeddings, embedding_dim})) {}
// num_embeddings(num_embeddings), embedding_dim(embedding_dim),
// weight(Tensor<dtype>({num_embeddings, embedding_dim})) {}

/**
 * using the following way, we can handle arbitrary dimension input.
 * @tparam dtype
 * input have shape (B, T), weight have shape (num_embeddings, embedding_dim)
 */
template <typename dtype>
Tensor<dtype> Embedding<dtype>::forward(const Tensor<dtype> &input) const {
  auto new_shape = input.shape();
  new_shape.push_back(this->embedding_dim);
  auto result = Tensor<dtype>(new_shape, input.device_type); // (B, T, embedding_dim)

  // std::vector<int> cur_idx(input.shape().size(), 0);

  // embedding every elements in input
  # pragma omp parallel for
  for (int i = 0; i < input.num_elements; i++) {
    auto cur_idx = input.getIndicesFromLinearIndex(i);
    int embedding_index = input.getData(cur_idx);

    auto a = weight.select(0, embedding_index); // (1, embedding_dim)
    auto a_new_shape = a.shape();
    a_new_shape.insert(a_new_shape.begin(), 1); // (1, 1, embedding_dim)
    a = a.view(a_new_shape);
    std::vector<std::vector<int>> slice = {{cur_idx[0], cur_idx[0]+1}, {cur_idx[1], cur_idx[1]+1}, {}};
    result.setItem(slice, a);
  }

  return result;
}

} // namespace nn
