#include "Transformer.hpp"
#include "Tensor.hpp"
#include "nn/modules.hpp"
#include <cmath>
#include <optional>
#include <vector>

template <typename dtype>
Attention<dtype>::Attention(ModelArgs args) {
    // this->n_kv_heads = (n_kv_heads == -1) ? n_heads : n_kv_heads;
    this->n_heads = args.n_heads;
    this->head_dim = args.dim / n_heads;

    this->wq = nn::Linear<dtype>(args.dim, args.dim);
    this->wk = nn::Linear<dtype>(args.dim, args.dim);
    this->wv = nn::Linear<dtype>(args.dim, args.dim);
    this->wo = nn::Linear<dtype>(args.dim, args.dim);

    this->cache_k = Tensor<dtype>({args.max_batch_size, args.max_seq_len, this->n_heads, this->head_dim});
    this->cache_v = Tensor<dtype>({args.max_batch_size, args.max_seq_len, this->n_heads, this->head_dim});
}

template <typename dtype>
Tensor<dtype> Attention<dtype>::forward(const Tensor<dtype>& x, int start_pos, const Tensor<dtype>& freqs, std::optional<Tensor<dtype>>& mask) {
    auto bsz = x.shape()[0];
    auto seqlen = x.shape()[1];

    auto xq = this->wq.forward(x);
    auto xk = this->wk.forward(x); // can we use kv cache to accept the result directly, saving the following step that copy result to cache?
    auto xv = this->wv.forward(x);

    xq = xq.view({bsz, seqlen, n_heads, head_dim});
    xk = xk.view({bsz, seqlen, n_heads, head_dim});
    xv = xv.view({bsz, seqlen, n_heads, head_dim});

    xq = apply_rotary_emb(xq, freqs, start_pos);
    xk = apply_rotary_emb(xk, freqs, start_pos);

    // put the computed k, v into kv_cache
    std::vector<std::vector<int>> slices  = {{0, bsz}, {start_pos, start_pos+seqlen}, {}, {}};
    this->cache_k.setItem(slices, xk);
    this->cache_v.setItem(slices, xv);

    // get keys and values from kv_cache
    std::vector<std::vector<int>> slices2  = {{0, bsz}, {0, start_pos+seqlen}, {}, {}};
    auto keys = this->cache_k.getItem(slices2);
    auto values = this->cache_v.getItem(slices2);

    xq = xq.transpose(1, 2);
    keys = keys.transpose(1, 2);
    values = values.transpose(1, 2); // (bsz, n_heads, cache_len+seqlen, head_dim)
    auto scores = xq.matmul(keys.transpose(2, 3)) / sqrt(head_dim); // (bsz, n_heads, seqlen, cache_len+seqlen)
    // std::cout << "scores: " << scores << std::endl;
    // std::cout << "mask: " << mask.value() << std::endl;
    if (mask.has_value()) {
        scores = scores + mask.value();
    }
    scores = scores.softmax(3); // (bsz, n_heads, seqlen, cache_len+seqlen)
    auto output = scores.matmul(values); // (bsz, n_heads, seqlen, head_dim)
    output = output.transpose(1, 2).contiguous().view({bsz, seqlen, n_heads * head_dim}); // (bsz, seqlen, dim)
    return this->wo.forward(output); // (bsz, seqlen, dim)
}

template <typename dtype>
FeedForward<dtype>::FeedForward(int dim, int hidden_dim) : dim(dim), hidden_dim(hidden_dim) {
    this->w1 = nn::Linear<dtype>(dim, hidden_dim);
    this->w2 = nn::Linear<dtype>(hidden_dim, dim);
    this->w3 = nn::Linear<dtype>(dim, hidden_dim);
}

template <typename dtype>
Tensor<dtype> FeedForward<dtype>::forward(const Tensor<dtype>& x) const {
    // x: (bsz, seqlen, dim)
    auto x1 = this->w1.forward(x); // (bsz, seqlen, hidden_dim)
    x1 = x1.silu();
    auto x3 = this->w3.forward(x); // (bsz, seqlen, hidden_dim)
    auto x2 = x1 * x3;
    auto result = this->w2.forward(x2); // (bsz, seqlen, dim)
    return result;
}


template class RMSNorm<float>;

template <typename dtype>
RMSNorm<dtype>::RMSNorm(int dim, float eps) : dim(dim), eps(eps) {
    // this->weight = Tensor<dtype>({dim});
    this->weight = randn<dtype>({dim});
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
    // std::cout << x << std::endl;
    // std::cout << weight << std::endl;
    // x : (bsz, seqlen, dim)
    // weight : (dim)
    auto result1 = this->_norm(x);
    // std::cout << result1 << std::endl;
    auto weight = this->weight.view({1, 1, this->dim});
    weight = weight.broadcast_to(x.shape());
    auto result2 = result1 * weight;

    // std::cout << result2 << std::endl;
    return result2;
}

template <typename dtype>
TransformerBlock<dtype>::TransformerBlock(int layer_id, ModelArgs args) {
    this->n_heads = args.n_heads;
    this->dim = args.dim;
    this->head_dim = args.dim / args.n_heads;
    this->layer_id = layer_id;

    this->attention = Attention<dtype>(args);
    this->feed_forward = FeedForward<dtype>(args.dim, args.hidden_dim);
    this->attention_norm = RMSNorm<dtype>(args.dim, 1e-5);
    this->ffn_norm = RMSNorm<dtype>(args.dim, 1e-5);
}

template <typename dtype>
Tensor<dtype> TransformerBlock<dtype>::forward(const Tensor<dtype>& x, int start_pos, const Tensor<dtype>& freqs, std::optional<Tensor<dtype>>& mask) {
    auto temp1 = this->attention_norm.forward(x);
    // std::cout << this->attention_norm.weight << std::endl;
    // std::cout << "temp1: " << temp1 << std::endl;
    auto h = x + this->attention.forward(temp1, start_pos, freqs, mask);
    auto out = h + this->feed_forward.forward(this->ffn_norm.forward(h));
    return out;
}

// Explicit instantiation for float, int
template class Transformer<float>;
// template class Transformer<int>;

// without default constructor, not initialize members in member initializer list cause error.
template <typename dtype>
Transformer<dtype>::Transformer(ModelArgs& args) {
    this->params = args;
    this->n_layers = args.n_layers;
    this->vocab_size = args.vocab_size;
    this->head_dim = args.dim / args.n_heads;
    this->tok_embeddings = nn::Embedding<dtype>(args.vocab_size, args.dim);
    // this->tok_embeddings = nn::Embedding<dtype>();
    this->layers = nn::ModuleList<dtype>();
    for (int i = 0; i < args.n_layers; i++) {
        // this->layers.append(TransformerBlock<dtype>(i, args));
        this->layers.append(std::make_shared<TransformerBlock<dtype>>(i, args));
    }
    this->norm = RMSNorm<dtype>(args.dim, 1e-5);
    this->output = nn::Linear<dtype>(args.dim, args.vocab_size);

    this->freqs = precompute_freqs();
}

template <typename dtype>
Tensor<dtype> Transformer<dtype>::precompute_freqs() {
    auto shape = {this->params.max_seq_len, this->head_dim}; // (seq_len, head_dim)
    Tensor<dtype> freqs(shape);
    for (int i = 0; i < this->params.max_seq_len; i++) {
        for (int j = 0; j < this->head_dim; j++) {
            freqs.setData({i, j},  i * pow(10000, -2.0 * j / this->head_dim));
        }
    }

    return freqs;
}

template <typename dtype>
std::optional<Tensor<dtype>> Transformer<dtype>::get_mask(int seqlen) const {
    if (seqlen <= 1) return {};

    Tensor<dtype> mask = Tensor<dtype>({seqlen, seqlen});
    for (int i = 0; i < seqlen; i++) {
        for (int j = 0; j < seqlen; j++) {
            if (i >= j) { // set diagonal to zero
                mask.setData({i, j}, 0);
            } else {
                mask.setData({i, j}, -INFINITY);
            }
        }
    }
    return mask;
}

template <typename dtype>
Tensor<dtype> Transformer<dtype>::forward(const Tensor<dtype>& tokens, int start_pos) const {
    auto bsz = tokens.shape()[0];
    auto seqlen = tokens.shape()[1];
    auto h = this->tok_embeddings.forward(tokens);
    // std::cout << h << std::endl;
    auto freqs = this->freqs.slice(start_pos, start_pos+seqlen, 0);
    auto mask = this->get_mask(seqlen);
    for (int i = 0; i < this->n_layers; i++) {
        auto layer = std::dynamic_pointer_cast<TransformerBlock<dtype>>(this->layers[i]);
        h = layer->forward(h, start_pos, freqs, mask);
        // std::cout << "layer " << i << " done" << std::endl;
        // std::cout << h << std::endl;
    }
    h = this->norm.forward(h);
    auto result = this->output.forward(h);
    return result;
}

