#include "Tensor.hpp"
#include "nn/modules.hpp"
#include <optional>

class ModelArgs {
public:
    int dim;
    int hidden_dim; // for FFN
    int n_layers;
    int n_heads;
    int vocab_size;
    // int n_kv_heads; (= n_heads), by default.

    int max_batch_size; // (=1), by default, we will process one batch at a time for simplicity.
    int max_seq_len;  // this is the sum of len(prompt) + len(generate)
};

template <typename dtype>
class Attention : public nn::Module<dtype>{
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

template <typename dtype>
class RMSNorm {
public:
    RMSNorm(int dim, float eps = 1e-5);

    Tensor<dtype> forward(Tensor<dtype> x);
    Tensor<dtype> _norm(Tensor<dtype> x);
private:
    float eps;
    int dim;
    Tensor<dtype> weight;
};

template <typename dtype>
class TransformerBlock {
public:
    TransformerBlock(int layer_id, ModelArgs args);

    Tensor<dtype> forward(Tensor<dtype> x, int start_pos, Tensor<dtype> freqs, std::optional<Tensor<dtype>> mask);
private:
    int n_heads;
    int dim;
    int head_dim;
    int layer_id;
    Attention<dtype> attention;
    FeedForward<dtype> feed_forward;
    RMSNorm<dtype> attention_norm, ffn_norm;
};

template <typename dtype>
class Transformer {
public:
    Transformer(ModelArgs args);

    Tensor<dtype> forward(Tensor<dtype> tokens, int start_pos);
private:
    int n_layers;
    int vocab_size;
    int head_dim;
    ModelArgs params;
    nn::Embedding<dtype> tok_embeddings;
    nn::ModuleList<dtype> layers;
    RMSNorm<dtype> norm;
    nn::Linear<dtype> output;
    Tensor<dtype> freqs;

    Tensor<dtype> precompute_freqs();
    std::optional<Tensor<dtype>> get_mask(int seq_len);
};
