#include "Tensor.hpp"
#include "nn/modules.hpp"
#include <optional>

class ModelArgs {
public:
    int dim;
    int n_layers;
    int n_heads;
    // int n_kv_heads; (= n_heads), by default.

    int max_batch_size; // (=1), by default, we will process one batch at a time for simplicity.
    int max_seq_len;  // this is the sum of len(prompt) + len(generate)
};

template <typename dtype>
class Attention {
public:
    Attention(ModelArgs args);

    Tensor<dtype> forward(Tensor<dtype> x, int start_pos, Tensor<dtype> freqs_cis, std::optional<Tensor<dtype>> mask);
private:
    // int n_kv_heads; // assum n_kn_heads == n_heads now for simplicity
    int n_heads;
    // int n_rep; // n_heads / n_kv_heads
    int head_dim;

    nn::Linear<dtype> wq, wk, wv, wo;
    Tensor<dtype> cache_k, cache_v;
};

template <typename dtype>
class FeedForward {
public:
    FeedForward(int dim, int hidden_dim); // no need multiple_of here, use llama2.c way.

    Tensor<dtype> forward(Tensor<dtype> x);
private:
    int dim, hidden_dim;
    nn::Linear<dtype> w1, w2, w3;
};
