#pragma once

#include "Tensor.hpp"
#include <cstddef>

// pre-allocated buffers, reduce tensor memory allocation and free.
template <typename dtype>
class ActivationBuffer {
public:
    ActivationBuffer(int batch_size, int seq_len, int dim, int hidden_dim, int n_heads, int max_seq_len, int vocab_size, std::string device_type) :
        x({batch_size, seq_len, dim}, device_type),
        x_norm({batch_size, seq_len, dim}, device_type),
        x_residual({batch_size, seq_len, dim}, device_type),
        xq({batch_size, seq_len, dim}, device_type),
        xk({batch_size, seq_len, dim}, device_type),
        xv({batch_size, seq_len, dim}, device_type),
        x1({batch_size, seq_len, hidden_dim}, device_type),
        x3({batch_size, seq_len, hidden_dim}, device_type),
        x2({batch_size, seq_len, hidden_dim}, device_type),
        scores({batch_size * n_heads * max_seq_len * max_seq_len}, device_type),
        scores2({batch_size * n_heads * max_seq_len * max_seq_len}, device_type),
        s_v({batch_size, n_heads, seq_len, dim / n_heads}, device_type),
        o_c({batch_size, seq_len, n_heads, dim / n_heads}, device_type),
        logical({batch_size, seq_len, vocab_size}, device_type),
        logical_v({batch_size, 1, vocab_size}, device_type),
        mask({batch_size * n_heads * seq_len * max_seq_len}, device_type),
        xq_c({batch_size, n_heads, seq_len, dim / n_heads}, device_type),
        keys_c({batch_size * max_seq_len * dim}, device_type),
        values_c({batch_size * max_seq_len * dim}, device_type),
        bsz(0), seq_len(0), dim(dim)
        // y(batch_size, seq_len, dim),
        // attn_out(batch_size, seq_len, dim),
        // mlp_out(batch_size, seq_len, dim),
        // attn_qkv(batch_size, seq_len, dim),
        // attn_q()
    {}

    Tensor<dtype> x;
    Tensor<dtype> x_norm;
    Tensor<dtype> x_residual;
    Tensor<dtype> xq, xk, xv;
    Tensor<dtype> x1, x3, x2;
    Tensor<dtype> scores;
    Tensor<dtype> scores2;
    Tensor<dtype> s_v;
    Tensor<dtype> o_c;
    Tensor<dtype> logical;
    Tensor<dtype> logical_v;
    Tensor<dtype> mask;
    Tensor<dtype> xq_c;
    Tensor<dtype> keys_c;
    Tensor<dtype> values_c;

    int bsz; // batch size
    int seq_len; // sequence length
    int dim; // dimension
};

