#pragma once

#include "Tensor.hpp"
#include <cstddef>

// pre-allocated buffers, reduce tensor memory allocation and free.
template <typename dtype>
class ActivationBuffer {
public:
    ActivationBuffer(int batch_size, int seq_len, int dim, int hidden_dim, int n_heads, int max_seq_len, std::string device_type) :
        x({batch_size, seq_len, dim}, device_type),
        x_norm({batch_size, seq_len, dim}, device_type),
        x_residual({batch_size, seq_len, dim}, device_type),
        xq({batch_size, seq_len, dim}, device_type),
        xk({batch_size, seq_len, dim}, device_type),
        xv({batch_size, seq_len, dim}, device_type),
        x1({batch_size, seq_len, hidden_dim}, device_type),
        x3({batch_size, seq_len, hidden_dim}, device_type),
        scores({batch_size * n_heads * max_seq_len * max_seq_len}, device_type),
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
    Tensor<dtype> x1, x3;
    Tensor<dtype> scores;

    int bsz; // batch size
    int seq_len; // sequence length
    int dim; // dimension
};

