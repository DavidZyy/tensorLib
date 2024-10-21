#pragma once

#include "Tokenizer.hpp"
#include "Transformer.hpp"

template <typename dtype>
class Llama2 {
public:
    Llama2(const std::string& model_path) : model(model_path) {
        // tokenizer = Tokenizer(model_path + "/tokenizer.json");
    }
    void generate();
    void text_completion();
private:
    Transformer<dtype> model;
    Tokenizer tokenizer;
};

