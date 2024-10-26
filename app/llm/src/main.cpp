#include "llama2.hpp"

const std::string prompt = "Once Upon a Time";
const std::string tokenizer_path = "./tokenizer.bin";

int main() {
    ModelArgs args;
    // args.dim = 512;
    args.dim = 16;
    // args.hidden_dim = 2048;
    args.hidden_dim = 32;
    // args.n_layers = 12;
    args.n_layers = 1;
    args.n_heads = 1;
    args.vocab_size = 32000;
    args.max_batch_size = 1; // set 1 first
    args.max_seq_len = 1024;

    auto generator = Llama2<float>(tokenizer_path, args);
    generator.text_completion(prompt);
}
