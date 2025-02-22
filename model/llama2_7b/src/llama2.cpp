#include "llama2.hpp"
#include "Tensor.hpp"
#include "Tokenizer.hpp"
#include <ostream>
#include <vector>
#include <chrono>
#include "llama2.hpp"

// if do not write below, will get error undefined reference to Llama2's methods
template class Llama2<float>;
template class Llama2<half>;
// template class Llama2<int>;
int total_len = 0;

template <typename dtype>
void Llama2<dtype>::generate(std::vector<int> prompt_tokens) {
    int prompt_len = prompt_tokens.size(); // the length of prompt tokens (prefill stage handles)
    int generate_len = 256; // the length of generation tokens (decode stage generates)

    Tensor<int> prompt_tokens_tensor({1, prompt_len}, this->device_type);  // shape: (bsz, seq_len)
    Tensor<int> next_token; // shape: (1, 1)
    Tensor<dtype> logits;
    std::chrono::duration<double> elapsed_time;
    int next_token_int;
    std::vector<std::vector<int>> slices;

    // copy prompt_tokens to tokens
    for (int i = 0; i < prompt_len; i++) {
        prompt_tokens_tensor.setData({0, i}, prompt_tokens[i]);
    }

/************************************* prefill stage ***********************************************************/

    auto start_time = std::chrono::high_resolution_clock::now();

    auto bsz = prompt_tokens_tensor.shape()[0];
    auto seqlen = prompt_tokens_tensor.shape()[1];

    ActivationBuffer<dtype> activation_buffer0(bsz, seqlen, this->model.params.dim, this->model.params.hidden_dim, this->device_type);

    logits = this->model.forward(prompt_tokens_tensor, total_len, activation_buffer0); // shape = (bsz, seq_len, vocab_size)
    total_len += prompt_len;
    slices = {{}, {logits.shape_[1]-1, logits.shape_[1]}, {}};
    logits = logits.getItem(slices); // (bsz, vocab_size), get the last token's logits
    next_token = logits.argmax(-1);
    next_token_int = next_token.getData({});

    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = end_time - start_time;
    std::cout << "(prefill speed: " << prompt_len / elapsed_time.count() << " tok/s)" << std::endl;

    std::cout << this->tokenizer.decode(-1, next_token_int) << std::flush; // use flush to output immediately, not cache in buffer

/************************************* decode stage ***********************************************************/

    start_time = std::chrono::high_resolution_clock::now();

    ActivationBuffer<dtype> activation_buffer1(bsz, 1, this->model.params.dim, this->model.params.hidden_dim, this->device_type);

    int generate_cnt = 0;
    for (int i = 0; i < generate_len; i++) {
        logits = this->model.forward(next_token, total_len, activation_buffer1);
        total_len += 1;
        next_token = logits.argmax(-1);
        next_token_int = next_token.getData({});

        generate_cnt++;
        if (next_token_int == 1) break;
        if (next_token_int == 2) { // EOS token
            break;
        }

        std::cout << this->tokenizer.decode(-1, next_token_int) << std::flush; // use flush to output immediately, not cache in buffer

    }

    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = end_time - start_time;
    std::cout << "(decode speed: " << generate_cnt / elapsed_time.count() << " tok/s)" << std::endl;
}

template <typename dtype>
void Llama2<dtype>::text_completion(const std::string& prompts) {
    std::vector<int> prompt_tokens = this->tokenizer.encode(prompts, true, false);
    generate(prompt_tokens);
}

void read_stdin(char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

template <typename dtype>
void Llama2<dtype>::chat() {
    int pos = 0;
    int steps = 256;

    // char user_prompt[512];
    // char user_prompt[] = "用中文讲一个故事";
    // char user_prompt[] = "tell me a story";
    char user_prompt[] = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. \
     If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.";
    char rendered_prompt[1152];
    char user_template[] = "[INST] %s [/INST]";
    // while (true) {
        // user input prompt
        std::cout<< "> ";
        // read_stdin(user_prompt, sizeof(user_prompt));
        sprintf(rendered_prompt, user_template, user_prompt);
    
        // convert char* to std::string
        std::string prompts(rendered_prompt);

        // assistant response
        std::vector<int> prompt_tokens = this->tokenizer.encode(prompts, true, false);
        // auto generation_tokens = generate(prompt_tokens);
        generate(prompt_tokens);
    // }
}
