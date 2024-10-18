#include "Transformer.hpp"
#include "Tensor.hpp"
#include "nn/modules.hpp"
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
