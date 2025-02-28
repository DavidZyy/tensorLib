#include "Transformer.hpp"
#include "Tensor.hpp"
#include "nn/modules.hpp"
#include "nn/embedding.hpp"
#include <cmath>
#include <optional>
#include <vector>
#include "Log.hpp"
#include "ActivationBuffer.hpp"

template <typename dtype>
Attention<dtype>::Attention(ModelArgs args, std::string device_type) : nn::Module<dtype>(device_type) {
    // this->n_kv_heads = (n_kv_heads == -1) ? n_heads : n_kv_heads;
    this->n_heads = args.n_heads;
    this->head_dim = args.dim / n_heads;

    this->wq = nn::Linear<dtype>(args.dim, args.dim, device_type);
    this->wk = nn::Linear<dtype>(args.dim, args.dim, device_type);
    this->wv = nn::Linear<dtype>(args.dim, args.dim, device_type);
    this->wo = nn::Linear<dtype>(args.dim, args.dim, device_type);

    this->cache_k = Tensor<dtype>({args.max_batch_size, args.max_seq_len, this->n_heads, this->head_dim}, device_type);
    this->cache_v = Tensor<dtype>({args.max_batch_size, args.max_seq_len, this->n_heads, this->head_dim}, device_type);
}

template <typename dtype>
// Tensor<dtype> Attention<dtype>::forward(const Tensor<dtype>& x, int start_pos, const Tensor<dtype>& freqs, std::optional<Tensor<dtype>>& mask) {
Tensor<dtype> Attention<dtype>::forward(ActivationBuffer<dtype>& activation_buffer, int start_pos, const Tensor<dtype>& freqs, std::optional<Tensor<dtype>>& mask) {
    auto x = activation_buffer.x_norm;

    auto bsz = activation_buffer.bsz;
    auto seqlen = activation_buffer.seq_len;

    // use kv cache to accept the result directly, saving the steps that copy result to kvcache.
    auto xq = this->wq.forward(x, activation_buffer.xq);
    auto xk = this->wk.forward(x, activation_buffer.xk);
    auto xv = this->wv.forward(x, activation_buffer.xv);

    xq = xq.view({bsz, seqlen, n_heads, head_dim});
    xk = xk.view({bsz, seqlen, n_heads, head_dim});
    xv = xv.view({bsz, seqlen, n_heads, head_dim});

    // xq = apply_rotary_emb(xq, start_pos, std::optional<Tensor<dtype>>(activation_buffer.xq)); // shape do not match
    // xk = apply_rotary_emb(xk, start_pos, std::optional<Tensor<dtype>>(activation_buffer.xk));
    // xq = apply_rotary_emb(xq, start_pos);
    // xk = apply_rotary_emb(xk, start_pos);
    xq = apply_rotary_emb(xq, start_pos, std::optional<Tensor<dtype>>(xq));
    xk = apply_rotary_emb(xk, start_pos, std::optional<Tensor<dtype>>(xk));

    // put the computed k, v into kv_cache
    std::vector<std::vector<int>> slices  = {{0, bsz}, {start_pos, start_pos+seqlen}, {}, {}};
    this->cache_k.setItem(slices, xk);
    this->cache_v.setItem(slices, xv);

    // get keys and values from kv_cache
    std::vector<std::vector<int>> slices2  = {{0, bsz}, {0, start_pos+seqlen}, {}, {}};
    auto keys = this->cache_k.getItem(slices2); // (bsz, cache_len+seqlen, n_heads, head_dim)
    auto values = this->cache_v.getItem(slices2); // (bsz, cache_len+seqlen, n_heads, head_dim)

    xq = xq.transpose(1, 2).contiguous(activation_buffer.xq_c); // (bsz, n_heads, seqlen, head_dim)
    keys = keys.transpose(1, 2); // (bsz, n_heads, cache_len+seqlen, head_dim)
    values = values.transpose(1, 2); // (bsz, n_heads, cache_len+seqlen, head_dim)

    // make keys and values are col major
/****************************************************************/
    std::vector<std::vector<int>> kv_size_slices  = {{0, (bsz * n_heads * (start_pos+seqlen) * head_dim)}};

    // std::vector<int> key_shape  = {bsz, n_heads, head_dim, start_pos+seqlen};
    std::vector<int> key_shape  = {bsz, n_heads, start_pos+seqlen, head_dim}; // make it col major (I know it is dirty here ...)
    auto keys_buffer = activation_buffer.keys_c.getItem(kv_size_slices);
    keys_buffer = keys_buffer.view(key_shape);
    keys = keys.contiguous(keys_buffer); // make it col major contiguous
    keys = keys.transpose(2, 3);;

    values = values.transpose(2, 3);
    std::vector<int> value_shape  = {bsz, n_heads, head_dim, start_pos+seqlen};
    auto values_buffer = activation_buffer.values_c.getItem(kv_size_slices);
    values_buffer = values_buffer.view(value_shape);
    values = values.contiguous(values_buffer); // (bsz, n_heads, cache_len+seqlen, head_dim)
    values = values.transpose(2, 3);
/***********************************************************************/

    std::vector<std::vector<int>> scores_size_slices  = {{0, (bsz * n_heads * seqlen * (start_pos+seqlen))}};
    std::vector<int> scores_shape  = {bsz, n_heads, seqlen, start_pos+seqlen};
    auto scores_buffer = activation_buffer.scores.getItem(scores_size_slices); // get a contiguous buffer for matmul
    scores_buffer = scores_buffer.view(scores_shape);

    auto scores_buffer_2 = activation_buffer.scores2.getItem(scores_size_slices); // get a contiguous buffer for matmul
    scores_buffer_2 = scores_buffer_2.view(scores_shape);

    // xq.matmul(keys.transpose(2, 3), scores_buffer); // (bsz, n_heads, seqlen, cache_len+seqlen)
    xq.matmul(keys, scores_buffer); // (bsz, n_heads, seqlen, cache_len+seqlen)
    // auto scores = scores_buffer / sqrt(head_dim);
    auto scores = scores_buffer.ScalarDiv(sqrt(head_dim), scores_buffer);

    // auto scores = xq.matmul(keys.transpose(2, 3)); // (bsz, n_heads, seqlen, cache_len+seqlen)
    // scores = scores / sqrt(head_dim);

    if (mask.has_value()) {
        // scores = scores + mask.value();
        scores = scores_buffer.EwiseAdd(mask.value(), scores_buffer);
    }

    // scores = scores.softmax(3); // (bsz, n_heads, seqlen, cache_len+seqlen)
    scores = scores.softmax(3, scores_buffer_2); // (bsz, n_heads, seqlen, cache_len+seqlen)
    // auto output = scores.matmul(values); // (bsz, n_heads, seqlen, head_dim)
    auto output = scores.matmul(values, activation_buffer.s_v); // (bsz, n_heads, seqlen, head_dim)
    // output = output.transpose(1, 2).contiguous().view({bsz, seqlen, n_heads * head_dim}); // (bsz, seqlen, dim)
    output = output.transpose(1, 2).contiguous(activation_buffer.o_c).view({bsz, seqlen, n_heads * head_dim}); // (bsz, seqlen, dim)
    // return this->wo.forward(output); // (bsz, seqlen, dim)
    return this->wo.forward(output, activation_buffer.x_norm); // (bsz, seqlen, dim)
}

template <typename dtype>
FeedForward<dtype>::FeedForward(int dim, int hidden_dim, std::string device_type) : nn::Module<dtype>(device_type), dim(dim), hidden_dim(hidden_dim) {
    this->w1 = nn::Linear<dtype>(dim, hidden_dim, device_type);
    this->w2 = nn::Linear<dtype>(hidden_dim, dim, device_type);
    this->w3 = nn::Linear<dtype>(dim, hidden_dim, device_type);
}

// template <typename dtype>
// Tensor<dtype> FeedForward<dtype>::forward(const Tensor<dtype>& x) const {
//     // x: (bsz, seqlen, dim)
//     auto x1 = this->w1.forward(x); // (bsz, seqlen, hidden_dim)
//     x1 = x1.silu();
//     auto x3 = this->w3.forward(x); // (bsz, seqlen, hidden_dim)
//     auto x2 = x1 * x3;
//     auto result = this->w2.forward(x2); // (bsz, seqlen, dim)
//     return result;
// }

template <typename dtype>
Tensor<dtype> FeedForward<dtype>::forward(ActivationBuffer<dtype>& activation_buffer) const {
    // x: (bsz, seqlen, dim)
    // auto x1 = this->w1.forward(x); // (bsz, seqlen, hidden_dim)
    auto x1 = this->w1.forward(activation_buffer.x_norm, activation_buffer.x1);
    x1 = x1.silu(activation_buffer.x1);
    // auto x3 = this->w3.forward(x); // (bsz, seqlen, hidden_dim)
    auto x3 = this->w3.forward(activation_buffer.x_norm, activation_buffer.x3);
    // auto x2 = x1 * x3;
    auto x2 = x1.EwiseMul(x3, activation_buffer.x2);
    // auto result = this->w2.forward(x2); // (bsz, seqlen, dim)
    auto result = this->w2.forward(x2, activation_buffer.x); // (bsz, seqlen, dim)
    return result;
}

template <typename dtype>
TransformerBlock<dtype>::TransformerBlock(int layer_id, ModelArgs args, std::string device_type) : nn::Module<dtype>(device_type) {
    this->n_heads = args.n_heads;
    this->dim = args.dim;
    this->head_dim = args.dim / args.n_heads;
    this->layer_id = layer_id;

    this->attention = Attention<dtype>(args, device_type);
    this->feed_forward = FeedForward<dtype>(args.dim, args.hidden_dim, device_type);
    this->attention_norm = nn::RMSNorm<dtype>(args.dim, 1e-5, device_type);
    this->ffn_norm = nn::RMSNorm<dtype>(args.dim, 1e-5, device_type);
}

template <typename dtype>
// Tensor<dtype> TransformerBlock<dtype>::forward(const Tensor<dtype>& x, int start_pos, const Tensor<dtype>& freqs, std::optional<Tensor<dtype>>& mask) {
Tensor<dtype> TransformerBlock<dtype>::forward(ActivationBuffer<dtype>& activation_buffer, int start_pos, const Tensor<dtype>& freqs, std::optional<Tensor<dtype>>& mask) {

    // auto temp1 = this->attention_norm.forward(x, x_norm);
    this->attention_norm.forward(activation_buffer.x, activation_buffer.x_norm);

    // auto h = x + this->attention.forward(x_norm, start_pos, freqs, mask);
    // auto h = activation_buffer.x + this->attention.forward(activation_buffer, start_pos, freqs, mask);
    auto h = activation_buffer.x.EwiseAdd( this->attention.forward(activation_buffer, start_pos, freqs, mask), activation_buffer.x_residual);

    this->ffn_norm.forward(h, activation_buffer.x_norm);

    // auto out = h + this->feed_forward.forward(activation_buffer.x_norm);
    // auto out = h + this->feed_forward.forward(activation_buffer);
    auto out = h.EwiseAdd(this->feed_forward.forward(activation_buffer), activation_buffer.x_residual);

    return out;
}

// Explicit instantiation for float, int
template class Transformer<half>;
template class Transformer<float>;

// without default constructor, not initialize members in member initializer list cause error.
template <typename dtype>
Transformer<dtype>::Transformer(ModelArgs& args, std::string device_type) : nn::Module<dtype>(device_type) {
    this->params = args;
    this->n_layers = args.n_layers;
    this->vocab_size = args.vocab_size;
    this->head_dim = args.dim / args.n_heads;
    this->tok_embeddings = nn::Embedding<dtype>(args.vocab_size, args.dim, device_type);
    // this->tok_embeddings = nn::Embedding<dtype>();
    this->layers = nn::ModuleList<dtype>(device_type);
    for (int i = 0; i < args.n_layers; i++) {
        // this->layers.append(TransformerBlock<dtype>(i, args));
        this->layers.append(std::make_shared<TransformerBlock<dtype>>(i, args, device_type));
    }
    this->norm = nn::RMSNorm<dtype>(args.dim, 1e-5, device_type);
    this->output = nn::Linear<dtype>(args.dim, args.vocab_size, device_type);

    this->freqs = precompute_freqs(); // no use
}

template <typename dtype>
Tensor<dtype> Transformer<dtype>::precompute_freqs() {
    auto shape = {this->params.max_seq_len, this->head_dim}; // (seq_len, head_dim)
    Tensor<dtype> freqs(shape, this->device_type);
    for (int i = 0; i < this->params.max_seq_len; i++) {
        for (int j = 0; j < this->head_dim; j++) {
            freqs.setData({i, j},  i * pow(10000, -2.0 * j / this->head_dim));
        }
    }

    return freqs;
}

/**
 * 
a example of mask: 
seq_len = 4
cache_len = 5

0 0 0 0 0  0 - - -
0 0 0 0 0  0 0 - -
0 0 0 0 0  0 0 0 -
0 0 0 0 0  0 0 0 0

start_pos == cache_len
 * @tparam dtype 
 */
template <typename dtype>
// std::optional<Tensor<dtype>> Transformer<dtype>::get_mask(int seqlen, int start_pos) const {
std::optional<Tensor<dtype>> Transformer<dtype>::get_mask(int seqlen, int start_pos, Tensor<dtype>& mask) const {
    if (seqlen <= 1) return {};

    // Tensor<dtype> mask = Tensor<dtype>({seqlen, seqlen + start_pos}, this->device_type);
    // mask.setItem({{}, {}},0);
    for (int m = 0; m < mask.shape()[0]; m++) {
        for (int n = 0; n < mask.shape()[1]; n++) {
            for (int i = 0; i < seqlen; i++) {
                for (int j = 0; j < seqlen + start_pos; j++) {
                    if (j > start_pos + i) { // Set upper triangular part to -INFINITY
                        if constexpr (std::is_same<dtype, __half>::value) {
                            mask.setData({m, n, i, j}, __float2half(-INFINITY)); // Convert -INFINITY to __half
                        } else {
                            mask.setData({m, n, i, j}, -INFINITY); // Use directly for float/double
                        }
                    } else { // Set lower triangular part to 0
                        if constexpr (std::is_same<dtype, __half>::value) {
                            mask.setData({m, n, i, j}, __float2half(0.0f)); // Convert 0 to __half
                        } else {
                            mask.setData({m, n, i, j}, 0); // Use directly for int/float/double
                        }
                    }
                }
            }
        }
    }

    return mask;
}

template <typename dtype>
Tensor<dtype> Transformer<dtype>::forward(const Tensor<int>& tokens, int start_pos, ActivationBuffer<dtype>& activation_buffer) const {
    auto bsz = tokens.shape()[0];
    auto seqlen = tokens.shape()[1];
    auto n_heads = this->params.n_heads;

    std::vector<std::vector<int>> slicess = {{0, bsz}, {0,seqlen}, {0,this->tok_embeddings.embedding_dim}};

    activation_buffer.bsz = bsz;
    activation_buffer.seq_len = seqlen;
    activation_buffer.dim = this->tok_embeddings.embedding_dim;
    
    // auto x = activation_buffer.x.getItem(slicess);
    this->tok_embeddings.forward(tokens, activation_buffer.x);

    auto freqs = this->freqs.slice(start_pos, start_pos+seqlen, 0);

    std::vector<std::vector<int>> mask_size_slices = {{0, (bsz * n_heads * seqlen * (start_pos+seqlen))}};
    std::vector<int> mask_shape = {bsz, n_heads, seqlen, start_pos+seqlen};
    auto mask_buffer = activation_buffer.mask.getItem(mask_size_slices);
    mask_buffer = mask_buffer.view(mask_shape);
    // auto mask = this->get_mask(seqlen, start_pos);
    auto mask = this->get_mask(seqlen, start_pos, mask_buffer);

    Tensor<dtype> h;
    for (int i = 0; i < this->n_layers; i++) {
        auto layer = std::dynamic_pointer_cast<TransformerBlock<dtype>>(this->layers[i]);
        h = layer->forward(activation_buffer, start_pos, freqs, mask); // (bsz, seq_len, dim)
        activation_buffer.x.setItem(slicess, h);
        // LOG_INFO("\n");
    }

    h = this->norm.forward(h, activation_buffer.x_norm);
    auto result = this->output.forward(h, activation_buffer.logical); // (bsz, seq_len, vocab_size)
    // LOG_INFO("\n\n");
    return result;
}
