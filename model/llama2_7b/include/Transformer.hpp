#pragma once
#include "Tensor.hpp"
#include "ActivationBuffer.hpp"
#include "nn/modules.hpp"
#include "nn/rmsNorm.hpp"
#include "nn/linear.hpp"
#include "nn/embedding.hpp"
#include "nn/container.hpp"
#include <optional>
#include <string>
#include "ActivationBuffer.hpp"

class ModelArgs {
public:
    int dim;
    int hidden_dim; // for FFN
    int n_layers;
    int n_heads;
    int vocab_size;
    int n_kv_heads; // (= n_heads), by default.

    int max_batch_size; // (=1), by default, we will process one batch at a time for simplicity.
    int max_seq_len;  // this is the sum of len(prompt) + len(generate)
};

template <typename dtype>
class Attention : public nn::Module<dtype> {
public:
    Attention() = default;
    Attention(ModelArgs args, std::string device_type = "cpu");

    // Tensor<dtype> forward(const Tensor<dtype>& x, int start_pos, const Tensor<dtype>& freqs, std::optional<Tensor<dtype>>& mask); // not const, cache will be modified
    Tensor<dtype> forward(ActivationBuffer<dtype>& activation_buffer, int start_pos, const Tensor<dtype>& freqs, std::optional<Tensor<dtype>>& mask); // not const, cache will be modified
// private:
    // int n_kv_heads; // assum n_kn_heads == n_heads now for simplicity
    int n_heads;
    // int n_rep; // n_heads / n_kv_heads
    int head_dim;

    nn::Linear<dtype> wq, wk, wv, wo;
    Tensor<dtype> cache_k, cache_v;
};

template <typename dtype>
class FeedForward : public nn::Module<dtype> {
public:
    FeedForward() = default;
    FeedForward(int dim, int hidden_dim, std::string device_type = "cpu"); // no need multiple_of here, use llama2.c way.

    // Tensor<dtype> forward(const Tensor<dtype>& x) const override;
    Tensor<dtype> forward(ActivationBuffer<dtype>& activation_buffer) const;
// private:
    int dim, hidden_dim;
    nn::Linear<dtype> w1, w2, w3;
};

template <typename dtype>
class TransformerBlock : public nn::Module<dtype> {
public:
    TransformerBlock() = default;
    TransformerBlock(int layer_id, ModelArgs args, std::string device_type = "cpu");

    // Tensor<dtype> forward(const Tensor<dtype>& x, int start_pos, const Tensor<dtype>& freqs, std::optional<Tensor<dtype>>& mask); //(not const, Attention.cache will be modified) cannot use override here, because the function signature(parameters) is different with the base module
    Tensor<dtype> forward(ActivationBuffer<dtype>& activation_buffer, int start_pos, const Tensor<dtype>& freqs, std::optional<Tensor<dtype>>& mask); //(not const, Attention.cache will be modified) cannot use override here, because the function signature(parameters) is different with the base module
// private:
    int n_heads;
    int dim;
    int head_dim;
    int layer_id;
    Attention<dtype> attention;
    FeedForward<dtype> feed_forward;
    nn::RMSNorm<dtype> attention_norm, ffn_norm;
};

template <typename dtype>
class Transformer : public nn::Module<dtype> {
public:
    Transformer(ModelArgs& args, std::string device_type = "cpu");

    Tensor<dtype> forward(const Tensor<int>& tokens, int start_pos, ActivationBuffer<dtype>& activation_buffer) const;
// private:
    int n_layers;
    int vocab_size;
    int head_dim;
    ModelArgs params;
    nn::Embedding<dtype> tok_embeddings;
    nn::ModuleList<dtype> layers;
    nn::RMSNorm<dtype> norm;
    nn::Linear<dtype> output;
    Tensor<dtype> freqs;

    Tensor<dtype> precompute_freqs();
    std::optional<Tensor<dtype>> get_mask(int seq_len, int start_pos) const;
};
