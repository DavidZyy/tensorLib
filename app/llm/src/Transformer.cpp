#include "Transformer.hpp"
#include "Tensor.hpp"
#include "nn/modules.hpp"

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
    auto xk = this->wk.forward(x);
    auto xv = this->wv.forward(x);

    xq = xq.view({bsz, seqlen, n_heads, head_dim});
    xk = xk.view({bsz, seqlen, n_heads, head_dim});
    xv = xv.view({bsz, seqlen, n_heads, head_dim});

    xq = apply_rotary_emb(xq, freqs);
    xk = apply_rotary_emb(xk, freqs);

}
