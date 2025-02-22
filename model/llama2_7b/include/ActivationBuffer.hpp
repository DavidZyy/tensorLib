#pragma once

#include "Tensor.hpp"

// pre-allocated buffers, reduce tensor memory allocation and free.
template <typename dtype>
class ActivationBuffer {
public:
    ActivationBuffer(int max_batch_size, int max_seq_len, int dim, std::string device_type) :
        x({max_batch_size, max_seq_len, dim}, device_type)
        // y(max_batch_size, max_seq_len, dim),
        // attn_out(max_batch_size, max_seq_len, dim),
        // mlp_out(max_batch_size, max_seq_len, dim),
        // attn_qkv(max_batch_size, max_seq_len, dim),
        // attn_q()
    {}

    Tensor<dtype> x;
};

