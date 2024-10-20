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
Tensor<dtype> Attention<dtype>::forward(Tensor<dtype> x, int start_pos, Tensor<dtype> freqs, std::optional<Tensor<dtype>> mask) {
    auto bsz = x.shape()[0];
    auto seqlen = x.shape()[1];

    auto xq = this->wq.forward(x);
    auto xk = this->wk.forward(x); // can we use kv cache to accept the result directly, saving the following step that copy result to cache?
    auto xv = this->wv.forward(x);

    xq = xq.view({bsz, seqlen, n_heads, head_dim});
    xk = xk.view({bsz, seqlen, n_heads, head_dim});
    xv = xv.view({bsz, seqlen, n_heads, head_dim});

    xq = apply_rotary_emb(xq, freqs);
    xk = apply_rotary_emb(xk, freqs);

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
Tensor<dtype> FeedForward<dtype>::forward(Tensor<dtype> x) {
    // x: (bsz, seqlen, dim)
    auto x1 = this->w1.forward(x); // (bsz, seqlen, hidden_dim)
    x1 = x1.silu();
    auto x3 = this->w3.forward(x); // (bsz, seqlen, hidden_dim)
    auto x2 = x1 * x3;
    auto result = this->w2.forward(x2); // (bsz, seqlen, dim)
    return result;
}


template <typename dtype>
RMSNorm<dtype>::RMSNorm(int dim, float eps) : dim(dim), eps(eps) {
    this->weight = Tensor<dtype>({dim});
}

template <typename dtype>
Tensor<dtype> RMSNorm<dtype>::_norm(Tensor<dtype> x) {
    auto origin_shape = x.shape();
    auto temp = x;
    temp = temp.pow(2);
    temp = temp.mean(-1, true);
    temp = temp.broadcast_to(origin_shape);
    temp = temp + this->eps;
    temp = temp.rsqrt();
    return x * temp; 
}

template <typename dtype>
Tensor<dtype> RMSNorm<dtype>::forward(Tensor<dtype> x) {
    // x : (bsz, seqlen, dim)
    // weight : (dim)
    auto result = this->_norm(x);
    auto weight = this->weight.view({1, 1, this->dim});
    weight = weight.broadcast_to(x.shape());
    return result * weight;
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
Tensor<dtype> TransformerBlock<dtype>::forward(Tensor<dtype> x, int start_pos, Tensor<dtype> freqs, std::optional<Tensor<dtype>> mask) { 
    auto temp1 = this->attention_norm.forward(x);
    auto h = x + this->attention.forward(temp1, start_pos, freqs, mask);
    auto out = h + this->feed_forward.forward(this->ffn_norm.forward(h));
    return out;
}


template <typename dtype>
Transformer<dtype>::Transformer(ModelArgs args) : params(args) {
    this->n_layers = args.n_layers;
    this->vocab_size = args.vocab_size;
    this->head_dim = args.dim / args.n_heads;
    this->tok_embeddings = nn::Embedding<dtype> (args.vocab_size, args.dim);
    this->layers = nn::ModuleList<dtype>();
    for (int i = 0; i < args.n_layers; i++) {
        this->layers.append(TransformerBlock<dtype>(i, args));
    }
    this->norm = RMSNorm<dtype>(args.dim, 1e-5);
    this->output = nn::Linear<dtype>(args.dim, args.vocab_size);

    this->freqs = precompute_freqs(args.dim);
}

template <typename dtype>
Tensor<dtype> Transformer<dtype>::precompute_freqs() {
    auto shape = {this->params.max_seq_len, this->head_dim}; // (seq_len, head_dim)
    Tensor<dtype> freqs(shape);
    for (int i = 0; i < this->params.max_seq_len; i++) {
        for (int j = 0; j < this->head_dim; j++) {
            freqs[i][j] = i * pow(10000, -2.0 * j / this->head_dim);
        }
    }

    return freqs;
}

template <typename dtype>
std::optional<Tensor<dtype>> Transformer<dtype>::get_mask(int seqlen) {
    if (seqlen <= 1) return {};

    Tensor<dtype> mask = Tensor<dtype>({seqlen, seqlen});
    for (int i = 0; i < seqlen; i++) {
        for (int j = 0; j < seqlen; j++) {
            if (i > j) {
                mask[i][j] = 0;
            } else {
                mask[i][j] = -INFINITY;
            }
        }
    }
    return mask;
}

template <typename dtype>
Tensor<dtype> Transformer<dtype>::forward(Tensor<dtype> tokens, int start_pos) {
    auto bsz = tokens.shape()[0];
    auto seqlen = tokens.shape()[1];
    auto h = this->tok_embeddings.forward(tokens);
    auto freqs = this->freqs.slice({start_pos, start_pos+seqlen});
    auto mask = this->get_mask(seqlen);
    for (int i = 0; i < this->n_layers; i++) {
        auto layer = this->layers[i];
        h = layer.forward(h, start_pos, freqs, mask);
    }
    h = this->norm.forward(h);
    output = this->output.forward(h);
    return output;
}

