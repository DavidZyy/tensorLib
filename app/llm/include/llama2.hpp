#pragma once

#include "Tensor.hpp"
#include "Tokenizer.hpp"
#include "Transformer.hpp"

template <typename dtype>
class Llama2 {
public:
    Llama2(const std::string& tokenizer_path, ModelArgs args) : model(args), tokenizer(tokenizer_path, 32000) {
    }
    Tensor<dtype> generate(std::vector<int> prompt_tokens);
    void text_completion(const std::string& prompts);
private:
    Transformer<dtype> model;
    Tokenizer tokenizer;
};

