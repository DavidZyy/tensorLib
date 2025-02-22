#pragma once

#include "Tensor.hpp"
#include "Tokenizer.hpp"
#include "Transformer.hpp"
#include "ActivationBuffer.hpp"
#include <string>

template <typename dtype>
class Llama2 {
public:
    Llama2(const std::string& tokenizer_path, ModelArgs args, std::string device_type) : 
        model(args, device_type),
        tokenizer(tokenizer_path, 32000),
        device_type(device_type)
        // activation_buffer(args.max_batch_size, args.max_seq_len, args.dim, device_type)
        {}

    void generate(std::vector<int> prompt_tokens);
    void text_completion(const std::string& prompts);
    void chat();

// private:
    Transformer<dtype> model;
    Tokenizer tokenizer;
    // ActivationBuffer<dtype> activation_buffer;
    std::string device_type;
};

