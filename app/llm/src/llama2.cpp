#include "llama2.hpp"
#include "Tensor.hpp"
#include "Tokenizer.hpp"
#include <ostream>
#include <vector>

// if do not write below, will get error undefined reference to Llama2's methods
template class Llama2<float>;
// template class Llama2<int>;

template <typename dtype>
Tensor<dtype> Llama2<dtype>::generate(std::vector<int> prompt_tokens) {
    int total_len = 1024;
    // int total_len = 256;
    // int total_len = 128;
    // int total_len = 32;
    Tensor<dtype> tokens({1, total_len}); // (bsz, max_seq_len)
    int prompt_len = prompt_tokens.size();

    // copy prompt_tokens to tokens
    for (int i = 0; i < prompt_len; i++) {
        tokens.data()[i] = prompt_tokens[i];
        std::cout << this->tokenizer.decode(-1, tokens.data()[i]) << std::flush;
    }

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    int prev_pos = 0;
    for (int cur_pos = prompt_len; cur_pos < total_len; cur_pos++) {
        std::vector<std::vector<int>> slices  = {{}, {prev_pos, cur_pos}};
        auto logits = model.forward(tokens.getItem(slices), prev_pos);  // logits.shape = (bsz, seq_len, vocab_size), seq_len = cur_pos - prev_pos
        slices = {{}, {logits.shape_[1]-1, logits.shape_[1]}, {}};
        logits = logits.getItem(slices); // (bsz, vocab_size), get the last of dim=1

        // std::cout << logits << std::endl;

        auto next_token = logits.argmax(-1); // (bsz, )
        if (next_token.data_[0] == 1) break;

        // std::cout << next_token.data_[0] << " " << std::flush;
        std::cout << this->tokenizer.decode(-1, next_token.data_[0]) << std::flush; // use flush to output immediately, not cache in buffer

        slices = {{}, {cur_pos, cur_pos+1}};
        tokens.setItem(slices, next_token);
        prev_pos = cur_pos;
    }

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in seconds
    std::chrono::duration<double> elapsed = end_time - start_time;
    // Calculate and display tok/s
    double tokens_per_second = (total_len - prompt_len) / elapsed.count();
    std::cout << "\nTokens per second: " << tokens_per_second << " tok/s" << std::endl;
    return tokens;
}

template <typename dtype>
void Llama2<dtype>::text_completion(const std::string& prompts) {
    std::vector<int> prompt_tokens = this->tokenizer.encode(prompts, true, false);
    auto generation_tokens = generate(prompt_tokens);
    // std::vector<int> generation_tokens_vec;
    // for (int i = 0; i < generation_tokens.shape_[1]; i++) {
    //     generation_tokens_vec.push_back(generation_tokens.data()[i]);
    // }
    // for (int i = 0; i < generation_tokens_vec.size(); i++) {
    //     int prev_token = i > 0 ? generation_tokens_vec[i-1] : -1;
    //     std::cout << this->tokenizer.decode(-1, generation_tokens_vec[i]) << " ";
    // }
}
